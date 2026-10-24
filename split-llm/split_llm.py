import argparse
import os
import json
import math
from collections import defaultdict

os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")

from dotenv import load_dotenv
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from safetensors.torch import save_file
try:
    from graphviz import Digraph
except ImportError:
    Digraph = None

load_dotenv()
    
MAX_SPLITS = 120
TARGET_SPLIT_SIZE_MB = 32

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt2", help="Model to split")
    parser.add_argument("--output_dir", type=str, default="splits_output", help="Base output directory")
    parser.add_argument("--max_n_splits", type=int, default=MAX_SPLITS, help="Maximum number of rows/buckets to create")
    parser.add_argument("--target_split_size_mb", type=float, default=TARGET_SPLIT_SIZE_MB,
                        help="Preferred size target for each split before capping by max_n_splits")
    parser.add_argument("--storage_strategy", choices=["separate", "merged"], default="separate",
                        help="Storage format: 'separate' (default) caps rows to max_n_splits and stores semantic layers as separate safetensors in different columns. 'merged' combines all layers of a bucket into a single safetensors file in column 1.")
    args = parser.parse_args()

    # Create a separate subfolder for this specific model
    model_folder_name = args.model.replace("/", "_").replace(":", "_")
    args.output_dir = os.path.join(args.output_dir, model_folder_name)
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading model {args.model}... this could take a while for large models as it downloads and loads ~50GB+ into memory.", flush=True)
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True)
    state_dict = model.state_dict()
    print("Model loaded successfully. Preparing splits...", flush=True)

    # Save enough local metadata to make the generated split directory self-sufficient.
    # The serving side should only need files from splits_output/<model>/.
    try:
        model.config.save_pretrained(args.output_dir)
    except Exception as exc:
        print(f"Warning: could not save model config locally: {exc}", flush=True)

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        tokenizer.save_pretrained(args.output_dir)
    except Exception as exc:
        print(f"Warning: could not save tokenizer locally: {exc}", flush=True)

    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None:
        try:
            generation_config.save_pretrained(args.output_dir)
        except Exception as exc:
            print(f"Warning: could not save generation config locally: {exc}", flush=True)

    manifest = {
        "model_name": args.model,
        "model_class": model.__class__.__name__,
        "config_class": model.config.__class__.__name__,
        "storage_strategy": args.storage_strategy,
        "max_n_splits": args.max_n_splits,
        "target_split_size_mb": args.target_split_size_mb,
        "total_parameters": sum(p.numel() for p in model.parameters()),
    }
    with open(os.path.join(args.output_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=3)
    
    groups = defaultdict(dict)
    for key, tensor in state_dict.items():
        parts = key.split('.')
        # Finer semantic grouping: use the immediate parent module.
        # e.g., 'model.layers.0.mlp.up_proj.weight' -> 'model.layers.0.mlp.up_proj'
        # This yields hundreds of fine-grained groups rather than just 48 top-level layer blocks.
        if len(parts) > 1:
             group_name = ".".join(parts[:-1])
        else:
             group_name = parts[0]
        groups[group_name][key] = tensor

    # Sort groups
    def sort_key(x):
        return [int(p) if p.isdigit() else p for p in x.split('.')]
    
    group_names = sorted(list(groups.keys()), key=sort_key)

    # Calculate byte size of each semantic group
    group_sizes = []
    total_size = 0
    for gn in group_names:
        size_bytes = sum(t.numel() * t.element_size() for t in groups[gn].values())
        group_sizes.append(size_bytes)
        total_size += size_bytes

    target_split_size_bytes = int(args.target_split_size_mb * 1024**2)
    max_rows = min(args.max_n_splits, max(1, math.ceil(total_size / target_split_size_bytes)))

    print(f"Balancing splits sizes into {max_rows} rows/buckets with a target of {args.target_split_size_mb:.2f} MB per split...")

    target_size = total_size / max_rows
    print(f"Actual average bucket size after capping: {target_size / 1024**2:.2f} MB per split.")

    # Pack groups with a lazy best-fit strategy so we do not create empty or near-empty buckets.
    bucket_records = []
    sorted_groups = sorted(zip(group_names, group_sizes), key=lambda item: item[1], reverse=True)

    for gn, size_bytes in sorted_groups:
        best_bucket_index = None
        best_bucket_size = None

        for idx, bucket in enumerate(bucket_records):
            projected_size = bucket["size"] + size_bytes
            if projected_size <= target_size:
                if best_bucket_size is None or projected_size > best_bucket_size:
                    best_bucket_index = idx
                    best_bucket_size = projected_size

        if best_bucket_index is None:
            if len(bucket_records) < max_rows:
                bucket_records.append({"groups": [gn], "size": size_bytes})
            else:
                bucket_index = min(range(len(bucket_records)), key=lambda idx: bucket_records[idx]["size"])
                bucket_records[bucket_index]["groups"].append(gn)
                bucket_records[bucket_index]["size"] += size_bytes
        else:
            bucket_records[best_bucket_index]["groups"].append(gn)
            bucket_records[best_bucket_index]["size"] += size_bytes

    min_bucket_size_bytes = int(args.target_split_size_mb * 0.5 * 1024**2)
    # Merge undersized buckets so tiny semantic fragments do not become standalone splits.
    while len(bucket_records) > 1:
        smallest_index = min(range(len(bucket_records)), key=lambda idx: bucket_records[idx]["size"])
        smallest_size = bucket_records[smallest_index]["size"]
        if smallest_size >= min_bucket_size_bytes:
            break

        merge_target = min(
            (idx for idx in range(len(bucket_records)) if idx != smallest_index),
            key=lambda idx: bucket_records[idx]["size"],
        )
        bucket_records[merge_target]["groups"].extend(bucket_records[smallest_index]["groups"])
        bucket_records[merge_target]["size"] += bucket_records[smallest_index]["size"]
        del bucket_records[smallest_index]

    # Keep each bucket's contents in model order for readability.
    group_order = {gn: idx for idx, gn in enumerate(group_names)}
    row_buckets = [
        sorted(bucket["groups"], key=lambda gn: group_order[gn])
        for bucket in bucket_records
    ]

    bucket_names = [f"split_{i}" for i in range(len(bucket_records))]

    allotments = []
    if Digraph:
        dot = Digraph(comment='Model Layout Scheme')
        dot.attr(rankdir='TB')
    else:
        dot = None
    
    prev_node = None
    base_dir_name = os.path.basename(args.output_dir)

    for i, (b_name, bucket_group_names) in tqdm(enumerate(zip(bucket_names, row_buckets)), total=len(bucket_names), desc="Saving splits"):
        meta_filename = f"{b_name}.json"
        
        # Calculate bucket size for PDF
        b_size_bytes = sum(group_sizes[group_order[gn]] for gn in bucket_group_names)
        if b_size_bytes >= 1024**3:
            size_str = f"{b_size_bytes / 1024**3:.2f} GB"
        else:
            size_str = f"{b_size_bytes / 1024**2:.2f} MB"
        
        if args.storage_strategy == "merged":
            # Combine all tensors for this bucket into one file
            merged_tensors = {}
            for gn in bucket_group_names:
                merged_tensors.update(groups[gn])
                
            st_filename = f"{b_name}.safetensors"
            st_filepath = os.path.join(args.output_dir, st_filename)
            save_file(merged_tensors, st_filepath)
            
            meta_filepath = os.path.join(args.output_dir, meta_filename)
            with open(meta_filepath, 'w') as f:
                json.dump({"bucket": b_name, "tensors": list(merged_tensors.keys()), "strategy": "merged"}, f, indent=3)
            
            allotments.append({
                "src": [f"{base_dir_name}/{meta_filename}", f"{base_dir_name}/{st_filename}"],
                "dst": [f"/{meta_filename}", f"/{st_filename}"],
                "row": i,
                "col": 0
            })
            
            if dot is not None:
                dot.node(str(i), f"{b_name}\n({len(merged_tensors)} tensors)\nSize: {size_str}", shape='box')

        else:
            # Separate strategy: Each semantic layer gets its own safetensors file in distinct columns (col >= 1)
            bucket_meta = {"bucket": b_name, "strategy": "separate", "layers": {}}
            bucket_srcs = [f"{base_dir_name}/{meta_filename}"]
            bucket_dsts = [f"/{meta_filename}"]
            total_tensors = 0
            
            for gn in bucket_group_names:
                layer_tensors = groups[gn]
                total_tensors += len(layer_tensors)
                
                st_filename = f"{gn.replace('.', '_')}.safetensors"
                st_filepath = os.path.join(args.output_dir, st_filename)
                save_file(layer_tensors, st_filepath)
                
                bucket_meta["layers"][gn] = list(layer_tensors.keys())
                
                bucket_srcs.append(f"{base_dir_name}/{st_filename}")
                bucket_dsts.append(f"/{st_filename}")
                
            meta_filepath = os.path.join(args.output_dir, meta_filename)
            with open(meta_filepath, 'w') as f:
                json.dump(bucket_meta, f, indent=3)

            allotments.append({
                "src": bucket_srcs,
                "dst": bucket_dsts,
                "row": i,
                "col": 0
            })
                
            if dot is not None:
                dot.node(str(i), f"{b_name}\n({len(bucket_group_names)} files, {total_tensors} tensors)\nSize: {size_str}", shape='box')
        
        if dot:
            if prev_node is not None:
                dot.edge(str(prev_node), str(i))
            prev_node = i
    
    field_path = os.path.join(args.output_dir, "field.json")
    with open(field_path, "w") as f:
        json.dump({"allotments": allotments}, f, indent=3)
        
    print(f"Created field.json with {len(bucket_names)} rows (buckets) and {len(allotments)} total allotments.")
    
    if dot:
        pdf_path = os.path.join(args.output_dir, "model_scheme")
        dot.render(pdf_path, format='pdf', cleanup=True)
        print(f"Created scheme PDF at {pdf_path}.pdf")
    else:
        print("Graphviz not found. PDF scheme not generated.")

if __name__ == '__main__':
    main()
