"""Run one or more model splits as a WebRTC relay pipeline.

The split artifacts in ``splits_output/<model>/`` are storage buckets produced
by ``split_llm.py``. This runner loads the requested split metadata and its
weight shards, then wires the selected splits into a sorted WebRTC chain.

The transport is real WebRTC via ``aiortc``. The default payload handler is a
safe relay that annotates each hop with split metadata while preserving the
payload contents. That keeps the script immediately usable and isolates the
model-specific execution hook in one place.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from aiortc import RTCPeerConnection
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import threading
import http.server
import socketserver
import urllib.parse as urlparse
import requests
from http import HTTPStatus




def sanitize_model_name(model_name: str) -> str:
	return model_name.replace("/", "_").replace(":", "_")


def parse_split_numbers(raw_values: Iterable[str]) -> list[int]:
	split_numbers: set[int] = set()
	for value in raw_values:
		for item in str(value).split(","):
			item = item.strip()
			if not item:
				continue
			split_numbers.add(int(item))
	return sorted(split_numbers)


def load_json(path: Path) -> dict[str, Any]:
	with path.open("r", encoding="utf-8") as handle:
		return json.load(handle)


def tensor_size_bytes(tensor: torch.Tensor) -> int:
	return tensor.numel() * tensor.element_size()


async def wait_for_channel_open(channel: Any) -> None:
	while channel.readyState != "open":
		await asyncio.sleep(0.01)


def encode_payload(payload: dict[str, Any]) -> str:
	return json.dumps(payload, separators=(",", ":"))


def decode_payload(message: Any) -> dict[str, Any]:
	if isinstance(message, bytes):
		message = message.decode("utf-8")
	if not isinstance(message, str):
		raise TypeError(f"Unsupported WebRTC payload type: {type(message)!r}")
	return json.loads(message)


def tensor_to_payload(tensor: torch.Tensor) -> dict[str, Any]:
	array = tensor.detach().cpu().contiguous().numpy()
	return {
		"kind": "tensor",
		"dtype": str(array.dtype),
		"shape": list(array.shape),
		"data_b64": base64.b64encode(array.tobytes()).decode("ascii"),
	}


def payload_to_tensor(payload: dict[str, Any]) -> torch.Tensor:
	dtype = np.dtype(payload["dtype"])
	raw = base64.b64decode(payload["data_b64"])
	array = np.frombuffer(raw, dtype=dtype).reshape(payload["shape"])
	return torch.from_numpy(array.copy())


_LOCAL_GENERATION_CACHE: dict[str, tuple[AutoTokenizer, AutoModelForCausalLM, torch.device]] = {}


def resolve_torch_device(raw_device: str) -> torch.device:
	if raw_device == "auto":
		raw_device = "cuda" if torch.cuda.is_available() else "cpu"

	device = torch.device(raw_device)
	if device.type == "cuda":
		if not torch.cuda.is_available():
			raise RuntimeError(f"CUDA device requested ({raw_device!r}), but CUDA is not available.")
		if device.index is not None:
			torch.cuda.set_device(device)

	return device


def load_safetensors_on_device(path: Path, device: torch.device) -> dict[str, torch.Tensor]:
	return load_file(str(path), device=str(device))


def load_local_generation_bundle(model_name: str, splits_root: Path, device: torch.device) -> tuple[AutoTokenizer, AutoModelForCausalLM, torch.device]:
	model_folder = splits_root / sanitize_model_name(model_name)
	cache_key = f"{model_folder.resolve()}::{device}"
	if cache_key in _LOCAL_GENERATION_CACHE:
		return _LOCAL_GENERATION_CACHE[cache_key]

	if not model_folder.exists():
		raise FileNotFoundError(f"Missing local split folder for model {model_name!r}: {model_folder}")

	tokenizer = AutoTokenizer.from_pretrained(model_folder, local_files_only=True, trust_remote_code=True)
	config = AutoConfig.from_pretrained(model_folder, local_files_only=True, trust_remote_code=True)

	state_dict: dict[str, torch.Tensor] = {}
	for shard_path in sorted(model_folder.glob("*.safetensors")):
		state_dict.update(load_safetensors_on_device(shard_path, device))

	try:
		model = AutoModelForCausalLM.from_pretrained(
			model_folder,
			local_files_only=True,
			trust_remote_code=True,
			state_dict=state_dict,
		)
	except Exception:
		model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
		model.load_state_dict(state_dict, strict=False)
		if hasattr(model, "tie_weights"):
			model.tie_weights()

	model.to(device)
	model.eval()
	_LOCAL_GENERATION_CACHE[cache_key] = (tokenizer, model, device)
	return tokenizer, model, device


def build_chat_reply(model_name: str, payload: dict[str, Any], stages: list[SplitStage]) -> str:
	text = str(payload.get("text", "")).strip()
	if not text and payload.get("kind") == "text":
		text = "(empty message)"

	if not stages:
		return text

	model_folder = stages[0].split_dir
	device = stages[0].device
	tokenizer, model, device = load_local_generation_bundle(model_name, model_folder.parent, device)

	messages = payload.get("messages")
	if isinstance(messages, list) and messages:
		prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
	elif getattr(tokenizer, "chat_template", None):
		prompt = tokenizer.apply_chat_template(
			[{"role": "user", "content": text}],
			tokenize=False,
			add_generation_prompt=True,
		)
	else:
		prompt = text

	inputs = tokenizer(prompt, return_tensors="pt")
	inputs = {key: value.to(device) for key, value in inputs.items()}
	input_length = inputs["input_ids"].shape[-1]

	with torch.inference_mode():
		generated = model.generate(
			**inputs,
			max_new_tokens=128,
			do_sample=False,
			pad_token_id=tokenizer.eos_token_id or tokenizer.pad_token_id,
		)

	new_tokens = generated[0][input_length:]
	response_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
	if response_text:
		return response_text

	return tokenizer.decode(generated[0], skip_special_tokens=True).strip() or text


@dataclass
class SplitStage:
	model_name: str
	split_number: int
	split_dir: Path
	metadata: dict[str, Any]
	device: torch.device = field(default_factory=lambda: torch.device("cpu"))
	shard_paths: list[Path] = field(default_factory=list)
	shards: dict[str, dict[str, torch.Tensor]] = field(default_factory=dict)
	output_channel: Any | None = None
	final_result: asyncio.Future | None = None

	@property
	def bucket_name(self) -> str:
		return self.metadata.get("bucket", f"split_{self.split_number}")

	@property
	def strategy(self) -> str:
		return self.metadata.get("strategy", "separate")

	def load_shards(self) -> None:
		if self.strategy == "merged":
			merged_path = self.split_dir / f"split_{self.split_number}.safetensors"
			if not merged_path.exists():
				raise FileNotFoundError(f"Missing merged split shard: {merged_path}")
			self.shard_paths = [merged_path]
			self.shards = {merged_path.name: load_safetensors_on_device(merged_path, self.device)}
			return

		layers = self.metadata.get("layers", {})
		if not layers:
			raise ValueError(f"Split {self.split_number} does not declare any layers")

		shard_paths: list[Path] = []
		shard_data: dict[str, dict[str, torch.Tensor]] = {}
		for layer_name in layers:
			shard_path = self.split_dir / f"{layer_name.replace('.', '_')}.safetensors"
			if not shard_path.exists():
				raise FileNotFoundError(f"Missing shard for layer {layer_name!r}: {shard_path}")
			shard_paths.append(shard_path)
			shard_data[shard_path.name] = load_safetensors_on_device(shard_path, self.device)

		self.shard_paths = shard_paths
		self.shards = shard_data

	@property
	def tensor_count(self) -> int:
		return sum(len(tensors) for tensors in self.shards.values())

	@property
	def tensor_bytes(self) -> int:
		return sum(tensor_size_bytes(tensor) for tensors in self.shards.values() for tensor in tensors.values())

	def summary(self) -> dict[str, Any]:
		return {
			"model": self.model_name,
			"split": self.split_number,
			"bucket": self.bucket_name,
			"strategy": self.strategy,
			"device": str(self.device),
			"tensor_count": self.tensor_count,
			"tensor_mb": round(self.tensor_bytes / 1024**2, 3),
			"shards": [path.name for path in self.shard_paths],
		}

	def process_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
		trace_entry = {
			"split": self.split_number,
			"bucket": self.bucket_name,
			"strategy": self.strategy,
			"device": str(self.device),
			"tensor_count": self.tensor_count,
			"tensor_mb": round(self.tensor_bytes / 1024**2, 3),
		}

		trace = list(payload.get("trace", []))
		trace.append(trace_entry)

		processed = dict(payload)
		processed["trace"] = trace
		processed["stage"] = self.summary()

		if processed.get("kind") == "tensor":
			tensor = payload_to_tensor(processed)
			processed["tensor_stats"] = {
				"shape": list(tensor.shape),
				"dtype": str(tensor.dtype),
				"norm": float(tensor.float().norm().item()) if tensor.numel() else 0.0,
			}

			# Attempt a minimal partial forward using local shards.
			# For each 2D weight found in the shards, if its input dim matches
			# the current tensor feature dim, apply a linear transform.
			try:
				x = tensor.to(self.device).float()
				applied = False
				for shard_name in (self.shard_paths or []):
					shard = self.shards.get(Path(shard_name).name, {})
					# apply weights then biases deterministically by key order
					for param_name in sorted(shard.keys()):
						param = shard[param_name].float()
						if param.dim() == 2:
							W = param
							# expect W shape (out, in)
							in_dim = W.shape[1]
							if x.dim() == 1 and x.shape[0] == in_dim:
								x = x @ W.t()
								applied = True
							elif x.dim() == 2 and x.shape[1] == in_dim:
								x = x @ W.t()
								applied = True
						elif param.dim() == 1:
							# bias
							b = param
							if applied:
								# add bias if shapes match
								if x.dim() == 1 and x.shape[0] == b.shape[0]:
									x = x + b
								elif x.dim() == 2 and x.shape[1] == b.shape[0]:
									x = x + b.unsqueeze(0)

				# if an op was applied, replace the payload with a tensor payload
				if applied:
					processed = tensor_to_payload(x)
					processed["stage"] = self.summary()
					processed["trace"] = trace
			except Exception:
				# don't fail the pipeline for execution errors; fall back to relay
				pass
		elif processed.get("kind") == "text":
			text = str(processed.get("text", ""))
			processed["text"] = text
			processed["text_stats"] = {
				"characters": len(text),
				"words": len(text.split()),
			}

		return processed


async def negotiate_pair(upstream: SplitStage, downstream: SplitStage) -> tuple[RTCPeerConnection, RTCPeerConnection, Any]:
	sender = RTCPeerConnection()
	receiver = RTCPeerConnection()
	ready = asyncio.Event()

	@receiver.on("datachannel")
	def on_datachannel(channel: Any) -> None:
		downstream.input_channel = channel

		@channel.on("message")
		def on_message(message: Any) -> None:
			asyncio.create_task(handle_stage_message(downstream, message))

		ready.set()

	channel = sender.createDataChannel(f"split-{upstream.split_number}-to-{downstream.split_number}")
	upstream.output_channel = channel

	offer = await sender.createOffer()
	await sender.setLocalDescription(offer)
	await receiver.setRemoteDescription(sender.localDescription)
	answer = await receiver.createAnswer()
	await receiver.setLocalDescription(answer)
	await sender.setRemoteDescription(receiver.localDescription)

	await ready.wait()
	return sender, receiver, channel


async def handle_stage_message(stage: SplitStage, message: Any) -> None:
	payload = decode_payload(message)
	processed = stage.process_payload(payload)

	if stage.output_channel is None:
		if stage.final_result is not None and not stage.final_result.done():
			stage.final_result.set_result(processed)
		return

	await wait_for_channel_open(stage.output_channel)
	stage.output_channel.send(encode_payload(processed))


async def run_pipeline(stages: list[SplitStage], initial_payload: dict[str, Any]) -> dict[str, Any]:
	pcs: list[RTCPeerConnection] = []
	channels: list[Any] = []

	for upstream, downstream in zip(stages[:-1], stages[1:]):
		sender_pc, receiver_pc, channel = await negotiate_pair(upstream, downstream)
		pcs.extend([sender_pc, receiver_pc])
		channels.append(channel)

	final_result: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
	stages[-1].final_result = final_result

	for channel in channels:
		await wait_for_channel_open(channel)

	await handle_stage_message(stages[0], encode_payload(initial_payload))
	result = await final_result

	for pc in pcs:
		await pc.close()

	return result


def build_stages(model_name: str, split_numbers: list[int], splits_root: Path, device: torch.device) -> list[SplitStage]:
	model_folder = splits_root / sanitize_model_name(model_name)
	if not model_folder.exists():
		raise FileNotFoundError(
			f"No split folder found for model {model_name!r}: {model_folder}"
		)

	stages: list[SplitStage] = []
	for split_number in split_numbers:
		metadata_path = model_folder / f"split_{split_number}.json"
		if not metadata_path.exists():
			raise FileNotFoundError(f"Missing split metadata: {metadata_path}")

		stage = SplitStage(
			model_name=model_name,
			split_number=split_number,
			split_dir=model_folder,
			metadata=load_json(metadata_path),
			device=device,
		)
		stage.load_shards()
		stages.append(stage)

	return stages


def print_plan(stages: list[SplitStage]) -> None:
	print("Planned WebRTC split pipeline:", flush=True)
	for index, stage in enumerate(stages, start=1):
		summary = stage.summary()
		print(
			f"  {index}. split {summary['split']} | bucket={summary['bucket']} | "
			f"strategy={summary['strategy']} | device={summary['device']} | tensors={summary['tensor_count']} | "
			f"size={summary['tensor_mb']} MB",
			flush=True,
		)


def build_initial_payload(args: argparse.Namespace) -> dict[str, Any]:
	if args.input_json:
		if Path(args.input_json).exists():
			return load_json(Path(args.input_json))
		return json.loads(args.input_json)

	if args.text is not None:
		return {"kind": "text", "text": args.text}

	stdin_text = sys.stdin.read().strip()
	if stdin_text:
		return {"kind": "text", "text": stdin_text}

	return {"kind": "text", "text": ""}


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Run split buckets through a WebRTC relay chain.")
	parser.add_argument("model_name", help="Model name used to resolve splits_output/<model>/")
	parser.add_argument("--mode", choices=["local","coordinator","worker"], default="local",
						help="Mode to run: 'local' (single-process pipeline), 'coordinator' (discovery service), or 'worker' (process owning splits)")
	parser.add_argument(
		"splits",
		nargs="+",
		help="One or more split numbers. Comma-separated values are also accepted.",
	)
	parser.add_argument(
		"--splits-output-dir",
		default="splits_output",
		help="Root directory that contains the model split folders.",
	)
	parser.add_argument(
		"--text",
		help="Optional text payload to inject at the start of the pipeline.",
	)
	parser.add_argument(
		"--input-json",
		help="Inline JSON payload or path to a JSON file to inject at the start of the pipeline.",
	)
	parser.add_argument("--host", default="0.0.0.0", help="Host address for coordinator or worker mode.")
	parser.add_argument("--port", default="0", help="Port for coordinator or worker mode.")
	parser.add_argument(
		"--coordinator",
		default=None,
		help="Coordinator base URL for worker registration and lookup.",
	)
	parser.add_argument(
		"--advertise-host",
		default=None,
		help="Reachable host/IP to advertise to the coordinator in worker mode. Defaults to --host, which is only safe when the bind host is directly reachable.",
	)
	parser.add_argument(
		"--device",
		default="cuda",
		help="Torch device for this process. Defaults to 'cuda'. Use 'auto', 'cpu', 'cuda:0', etc. In worker mode this binds the owned splits to that device.",
	)
	return parser


class CoordinatorHandler(http.server.BaseHTTPRequestHandler):
	registry: dict[int, tuple[str,int]] = {}

	def _send_json(self, obj, code=200):
		data = json.dumps(obj).encode("utf-8")
		self.send_response(code)
		self.send_header("Content-Type", "application/json")
		self.send_header("Content-Length", str(len(data)))
		self.end_headers()
		self.wfile.write(data)

	def do_POST(self):
		length = int(self.headers.get("Content-Length", 0))
		body = self.rfile.read(length) if length else b""
		try:
			payload = json.loads(body.decode("utf-8") or "{}")
		except Exception:
			return self._send_json({"error": "invalid json"}, code=400)

		if self.path == "/register":
			host = payload.get("host")
			port = int(payload.get("port"))
			splits = payload.get("splits", [])
			for s in splits:
				CoordinatorHandler.registry[int(s)] = (host, int(port))
			return self._send_json({"ok": True, "registered": splits})

		if self.path == "/unregister":
			host = payload.get("host")
			port = int(payload.get("port"))
			to_remove = [k for k,v in CoordinatorHandler.registry.items() if v == (host, port)]
			for k in to_remove:
				del CoordinatorHandler.registry[k]
			return self._send_json({"ok": True, "removed": to_remove})

		return self._send_json({"error": "unknown path"}, code=404)

	def do_GET(self):
		parsed = urlparse.urlparse(self.path)
		if parsed.path == "/lookup":
			q = urlparse.parse_qs(parsed.query)
			split = q.get("split")
			if not split:
				return self._send_json({"error": "missing split"}, code=400)
			split = int(split[0])
			val = CoordinatorHandler.registry.get(split)
			if not val:
				return self._send_json({"error": "unknown split"}, code=404)
			host, port = val
			return self._send_json({"host": host, "port": port})
		return self._send_json({"error": "unknown path"}, code=404)


def run_coordinator(host: str, port: int) -> None:
	with socketserver.TCPServer((host, port), CoordinatorHandler) as httpd:
		print(f"Coordinator serving on {host}:{port}")
		try:
			httpd.serve_forever()
		except KeyboardInterrupt:
			print("Coordinator stopped")


def parse_field_rows(model_folder: Path) -> tuple[dict[int,int], dict[int,list[int]]]:
	# returns split->row and row->list[splits]
	field_path = model_folder / "field.json"
	if not field_path.exists():
		return {}, {}
	data = load_json(field_path)
	allotments = data.get("allotments", [])
	split_to_row: dict[int,int] = {}
	row_to_splits: dict[int,list[int]] = {}
	for a in allotments:
		srcs = a.get("src", [])
		row = int(a.get("row", -1))
		for src in srcs:
			name = Path(src).name
			if name.startswith("split_") and name.endswith(".json"):
				try:
					n = int(name[len("split_"):-5])
				except Exception:
					continue
				split_to_row[n] = row
				row_to_splits.setdefault(row, []).append(n)
	return split_to_row, row_to_splits


class WorkerHTTPRequestHandler(http.server.BaseHTTPRequestHandler):
	server_ref = None

	def _send_json(self, obj, code=200):
		data = json.dumps(obj).encode("utf-8")
		self.send_response(code)
		self.send_header("Content-Type", "application/json")
		self.send_header("Content-Length", str(len(data)))
		self.end_headers()
		self.wfile.write(data)

	def do_POST(self):
		length = int(self.headers.get("Content-Length", 0))
		body = self.rfile.read(length) if length else b""
		try:
			payload = json.loads(body.decode("utf-8") or "{}")
		except Exception:
			return self._send_json({"error": "invalid json"}, code=400)

		if self.path == "/invoke":
			result = self.server_ref.handle_invoke(payload)
			return self._send_json({"ok": True, "result": result})

		return self._send_json({"error": "unknown path"}, code=404)


class WorkerHTTPServer(threading.Thread):
	def __init__(self, host: str, port: int, stages: list[SplitStage], model_name: str, splits_root: Path, coordinator: str | None, advertise_host: str | None = None):
		super().__init__(daemon=True)
		self.host = host
		self.port = port
		self.stages = stages
		self.model_name = model_name
		self.splits_root = splits_root
		self.coordinator = coordinator
		self.advertise_host = advertise_host or host
		self.httpd = None
		self._stop = threading.Event()

	def start(self):
		handler = WorkerHTTPRequestHandler
		handler.server_ref = self
		self.httpd = socketserver.TCPServer((self.host, self.port), handler)
		sa = self.httpd.socket.getsockname()
		self.port = sa[1]
		# register if coordinator provided
		if self.coordinator:
			url = self.coordinator.rstrip("/") + "/register"
			payload = {"host": self.advertise_host, "port": self.port, "splits": [s.split_number for s in self.stages]}
			try:
				resp = requests.post(url, json=payload, timeout=5)
				print("Registered with coordinator:", resp.status_code, resp.text)
			except Exception as e:
				print("Coordinator registration failed:", e)

		super().start()
		print(f"Worker listening on {self.host}:{self.port}")

	def run(self):
		try:
			while not self._stop.is_set():
				self.httpd.handle_request()
		except Exception:
			pass

	def stop(self):
		# unregister
		if self.coordinator:
			url = self.coordinator.rstrip("/") + "/unregister"
			payload = {"host": self.advertise_host, "port": self.port}
			try:
				requests.post(url, json=payload, timeout=3)
			except Exception:
				pass
		self._stop.set()
		if self.httpd:
			try:
				self.httpd.server_close()
			except Exception:
				pass

	def join(self, timeout: float | None = None) -> None:
		try:
			while not self._stop.is_set():
				self._stop.wait(1.0)
		except KeyboardInterrupt:
			self.stop()

	def handle_invoke(self, payload: dict[str, Any]) -> dict[str, Any]:
		# Process payload through local stages sequentially
		current = payload
		for stage in self.stages:
			current = stage.process_payload(current)

		# determine next row using field.json
		model_folder = self.splits_root / sanitize_model_name(self.model_name)
		split_to_row, row_to_splits = parse_field_rows(model_folder)
		max_row = max(row_to_splits) if row_to_splits else max(split_to_row.values(), default=-1)
		# find max local row
		local_rows = [split_to_row.get(s.split_number, None) for s in self.stages]
		local_rows = [r for r in local_rows if r is not None]
		if not local_rows:
			current["response_text"] = build_chat_reply(self.model_name, current, self.stages)
			current["assistant_message"] = {"role": "assistant", "content": current["response_text"]}
			print("No row info; final result:\n", json.dumps(current, indent=2))
			return current
		max_local = max(local_rows)
		if max_local >= max_row:
			current["response_text"] = build_chat_reply(self.model_name, current, self.stages)
			current["assistant_message"] = {"role": "assistant", "content": current["response_text"]}
			print("Reached final row; final result:\n", json.dumps(current, indent=2))
			return current
		next_row = max_local + 1
		next_splits = row_to_splits.get(next_row, [])
		if not next_splits:
			current["response_text"] = build_chat_reply(self.model_name, current, self.stages)
			current["assistant_message"] = {"role": "assistant", "content": current["response_text"]}
			print("Reached final row; final result:\n", json.dumps(current, indent=2))
			return current

		# lookup owners and forward
		for ns in next_splits:
			owner = None
			if self.coordinator:
				try:
					qurl = self.coordinator.rstrip("/") + f"/lookup?split={ns}"
					r = requests.get(qurl, timeout=5)
					if r.status_code == 200:
						j = r.json()
						owner = (j.get("host"), int(j.get("port")))
				except Exception as e:
					print("Lookup failed for split", ns, e)

			if owner:
				host, port = owner
				try:
					url = f"http://{host}:{port}/invoke"
					resp = requests.post(url, json=current, timeout=20)
					resp.raise_for_status()
					forwarded = resp.json()
					print(f"Forwarded to {host}:{port} for split {ns}")
					return forwarded.get("result", forwarded)
				except Exception as e:
					print("Forward failed:", e)
			else:
				print(f"No owner found for split {ns}; skipping forward")

		return current


async def async_main(args: argparse.Namespace) -> int:
	split_numbers = parse_split_numbers(args.splits)
	if not split_numbers:
		raise ValueError("At least one split number is required.")

	mode = getattr(args, "mode", "local")

	if mode == "coordinator":
		# Run a simple coordinator HTTP server for registration and lookups
		host = getattr(args, "host", "0.0.0.0")
		port = int(getattr(args, "port", 9000))
		print(f"Starting coordinator on {host}:{port}")
		run_coordinator(host, port)
		return 0

	device = resolve_torch_device(getattr(args, "device", "auto"))

	if mode == "worker":
		# Worker mode: register with coordinator and run HTTP server
		split_numbers = parse_split_numbers(args.splits)
		if not split_numbers:
			raise ValueError("At least one split number is required in worker mode.")

		host = getattr(args, "host", "0.0.0.0")
		port = int(getattr(args, "port", 0) or 8000)
		coordinator = getattr(args, "coordinator", None)
		advertise_host = getattr(args, "advertise_host", None) or host

		stages = build_stages(args.model_name, split_numbers, Path(args.splits_output_dir), device)
		for s in stages:
			print("Owned:", s.summary())

		server = WorkerHTTPServer(host, port, stages, args.model_name, Path(args.splits_output_dir), coordinator, advertise_host=advertise_host)
		server.start()
		try:
			server.join()
		except KeyboardInterrupt:
			server.stop()
		return 0

	stages = build_stages(args.model_name, split_numbers, Path(args.splits_output_dir), device)
	print_plan(stages)

	payload = build_initial_payload(args)
	print("Starting WebRTC chain...", flush=True)
	final_payload = await run_pipeline(stages, payload)
	if final_payload.get("kind") == "text":
		final_payload["response_text"] = build_chat_reply(args.model_name, final_payload, stages)
		final_payload["assistant_message"] = {"role": "assistant", "content": final_payload["response_text"]}

	print("Pipeline finished. Final payload:", flush=True)
	print(json.dumps(final_payload, indent=2, sort_keys=True), flush=True)
	return 0


def main() -> int:
	parser = build_parser()
	args = parser.parse_args()

	try:
		return asyncio.run(async_main(args))
	except KeyboardInterrupt:
		return 130


if __name__ == "__main__":
	raise SystemExit(main())
