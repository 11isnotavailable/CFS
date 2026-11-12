"""Evaluate causal-LM loss on every token of a frozen context collection."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".cache" / "huggingface"))

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--contexts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--logit-chunk", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    ).eval().to(args.device)
    model.requires_grad_(False)
    contexts = np.load(args.contexts, allow_pickle=False)["input_ids"][: args.num_samples]
    loss_sum, token_count = 0.0, 0
    for start in range(0, len(contexts), args.batch_size):
        ids = torch.from_numpy(contexts[start : start + args.batch_size].astype(np.int64)).to(args.device)
        with torch.inference_mode():
            hidden = model.model(input_ids=ids, use_cache=False, return_dict=True).last_hidden_state
            for position in range(0, ids.shape[1] - 1, args.logit_chunk):
                stop = min(position + args.logit_chunk, ids.shape[1] - 1)
                logits = model.lm_head(hidden[:, position:stop]).float()
                targets = ids[:, position + 1 : stop + 1]
                loss_sum += float(F.cross_entropy(
                    logits.flatten(0, 1), targets.flatten(), reduction="sum"
                ))
                token_count += targets.numel()
        print(f"samples={min(start + args.batch_size, len(contexts))}/{len(contexts)}", flush=True)
    loss = loss_sum / token_count
    result = {
        "model": args.model,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "frozen_contexts": str(args.contexts),
        "samples": len(contexts),
        "sequence_length": int(contexts.shape[1] - 1),
        "tokens": token_count,
        "loss": loss,
        "perplexity": math.exp(loss),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
