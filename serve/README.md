# Distributed Split Runner

This directory contains the runner, coordinator, worker scripts, and the commands you can use to drive a split model pipeline.

## Arguments

`distributed_serve.py` expects these arguments:

- `model_name`: the original model identifier, such as `openai-community/gpt2` or `google/medgemma-27b-it`. The script maps it to `splits_output/<model>/` by replacing `/` with `_`.
- `splits`: one or more split numbers to load. You can pass values like `0 1` or a comma-separated value like `0,1`.
- `--mode`: selects the runtime mode. `local` runs a single-process pipeline, `coordinator` starts the discovery service, and `worker` starts a split owner process.
- `--splits-output-dir`: root directory that contains the split folders. It defaults to `splits_output`.
- `--text`: injects a text payload at the start of the pipeline.
- `--input-json`: injects JSON directly, or points at a JSON file to load.
- `--host`: bind address for coordinator or worker mode.
- `--port`: listen port for coordinator or worker mode.
- `--coordinator`: coordinator URL used by worker mode for registration and owner lookup.
- `--advertise-host`: reachable host or IP that the worker advertises to the coordinator. Use this when the bind host is `0.0.0.0` or when processes run on different machines.
- `--device`: Torch device for local or worker execution. It defaults to `cuda`; use `cuda:0`, `cuda:1`, `cpu`, or `auto` as needed.

Each model directory under `splits_output/<model>/` is self-contained after regeneration. It includes `field.json`, the per-split `split_*.json` files, the split `*.safetensors` files, plus local `config.json`, tokenizer files, optional generation config, and `manifest.json`.

## Layout

- `distributed_serve.py`: main runner and worker/coordinator entrypoint
- `scripts/run_local.sh`: run a single-process local pipeline
- `scripts/run_coord.sh`: start the coordinator service
- `scripts/run_worker.sh`: start a worker and register it with the coordinator
- `scripts/invoke_first.sh`: send the first payload to a worker

## Local run

From the repository root:

```bash
./serve/scripts/run_local.sh openai-community/gpt2 0 1 --text "hello"
```

If you want to run several processes on the same machine, start one coordinator and one worker per split group on different ports. The first worker should receive the first range of splits, and the second worker should receive the remaining range.

```bash
# terminal 1
./serve/scripts/run_coord.sh openai-community/gpt2 127.0.0.1 9000

# terminal 2
./serve/scripts/run_worker.sh openai-community/gpt2 0,1,2,3,4,5,6,7 127.0.0.1 8001 http://127.0.0.1:9000 127.0.0.1

# terminal 3
./serve/scripts/run_worker.sh openai-community/gpt2 8,9,10,11,12,13,14 127.0.0.1 8002 http://127.0.0.1:9000 127.0.0.1

# terminal 4
./serve/scripts/invoke_first.sh 127.0.0.1 8001 "hello"
```

## MedGemma local multi-process

The `google/medgemma-27b-it` split folder is present in this workspace. You can run it on one machine with two workers the same way:

```bash
# terminal 1
./serve/scripts/run_coord.sh google/medgemma-27b-it 127.0.0.1 9010

# terminal 2
./serve/scripts/run_worker.sh google/medgemma-27b-it "$(seq -s, 0 59)" 127.0.0.1 8011 http://127.0.0.1:9010 127.0.0.1


# terminal 3
./serve/scripts/run_worker.sh google/medgemma-27b-it "$(seq -s, 60 119)" 127.0.0.1 8012 http://127.0.0.1:9010 127.0.0.1


# terminal 4
./serve/scripts/invoke_first.sh 127.0.0.1 8011 "hello. who are you?"
```

`run_worker.sh` accepts an optional seventh argument for the device:

```bash
./serve/scripts/run_worker.sh google/medgemma-27b-it "$(seq -s, 0 59)" 127.0.0.1 8011 http://127.0.0.1:9010 127.0.0.1 cuda:0
./serve/scripts/run_worker.sh google/medgemma-27b-it "$(seq -s, 60 119)" 127.0.0.1 8012 http://127.0.0.1:9010 127.0.0.1 cuda:1
```

If PyTorch reports that the NVIDIA driver is too old, CUDA will not be available to the worker until the driver is updated or the `kras` environment uses a PyTorch build compiled for that driver. For a CPU fallback while you fix the driver/toolkit mismatch, pass `cpu` as the seventh argument.

## Coordinator

Start the coordinator first:

```bash
./serve/scripts/run_coord.sh openai-community/gpt2 0.0.0.0 9000
```

## Workers

Start one worker per split group. Pass a reachable `advertise_host` when the worker is on a different machine from the coordinator or from the caller.

```bash
./serve/scripts/run_worker.sh openai-community/gpt2 0 0.0.0.0 8001 http://10.0.0.10:9000 10.0.0.21
./serve/scripts/run_worker.sh openai-community/gpt2 1 0.0.0.0 8002 http://10.0.0.10:9000 10.0.0.22
```

## Medgemma example

The workspace includes `google_medgemma-27b-it` splits. The example below uses the same two-worker pattern, but with shorter split lists shown as a compact alternative:

```bash
./serve/scripts/run_coord.sh google/medgemma-27b-it 127.0.0.1 9010
./serve/scripts/run_worker.sh google/medgemma-27b-it "$(seq -s, 0 59)" 127.0.0.1 8011 http://127.0.0.1:9010 127.0.0.1
./serve/scripts/run_worker.sh google/medgemma-27b-it "$(seq -s, 60 119)" 127.0.0.1 8012 http://127.0.0.1:9010 127.0.0.1
./serve/scripts/invoke_first.sh 127.0.0.1 8011 "hello"
```

## Kick off the pipeline

Send the first payload to the first worker. Use the worker's reachable address, not `0.0.0.0`. The worker returns the final payload in the HTTP response.

The last worker in the chain is terminal. It should return the final payload and stop, even if there is no next owner to forward to.

```bash
./serve/scripts/invoke_first.sh 127.0.0.1 8001 "hello"
```

## Different machines

If the coordinator and workers are on different machines, use the same pattern with real network addresses:

```bash
# machine A
./serve/scripts/run_coord.sh openai-community/gpt2 0.0.0.0 9000

# machine B
./serve/scripts/run_worker.sh openai-community/gpt2 0 0.0.0.0 8001 http://10.0.0.10:9000 10.0.0.21

# machine C
./serve/scripts/run_worker.sh openai-community/gpt2 1 0.0.0.0 8002 http://10.0.0.10:9000 10.0.0.22

# any machine that can reach the first worker
./serve/scripts/invoke_first.sh 10.0.0.21 8001 "hello"
```

## Notes

- The scripts default to the `splitllm-gpu` conda environment. Override with `SPLIT_LLM_CONDA_ENV=<name>` if needed.
- Use comma-separated split lists if a worker should own multiple splits.
- `distributed_serve.py` supports `--mode local`, `--mode coordinator`, and `--mode worker`.
- `run_worker.sh` device defaults to `cuda`; pass `cpu`, `auto`, `cuda:0`, etc. as the seventh argument to override it.
