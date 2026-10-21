"""Read-only CPU checks of catalog reachability and sampled SFT weight updates.

Loads local tokenizer files and small safetensors slices; no model download or training.
"""

import argparse
import ast
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def source_constraints(root, prefix_ids, tokenizer, prefix_index):
    """Execute only the actual evaluator's hash builder and callback, without loading its model."""
    tree = ast.parse((root / "evaluate.py").read_text())
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    get_hash = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "get_hash")

    def assigns(node, name):
        return isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)

    start = next(i for i, node in enumerate(main.body) if assigns(node, "hash_dict"))
    end = next(i for i, node in enumerate(main.body) if assigns(node, "hash_dict_title"))
    callback = next(node for node in main.body if isinstance(node, ast.FunctionDef)
                    and node.name == "prefix_allowed_tokens_fn_semantic")
    conversion = next(node for node in main.body if isinstance(node, ast.For)
                      and ast.unparse(node.iter) == "hash_dict.keys()")
    namespace = {"prefixID": [list(ids) for ids in prefix_ids], "tokenizer": tokenizer,
                 "prefix_index": prefix_index}
    code = ast.Module(body=[get_hash, *main.body[start:end], conversion, callback], type_ignores=[])
    exec(compile(code, str(root / "evaluate.py"), "exec"), namespace)
    return namespace["prefix_allowed_tokens_fn_semantic"]


def check_decoder(root, checkpoint, catalog, torch):
    from transformers import AutoConfig, AutoTokenizer
    sys.path.insert(0, str(root))
    from LogitProcessor import ConstrainedLogitsProcessor

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    config = AutoConfig.from_pretrained(checkpoint, local_files_only=True)
    if "llama" in str(checkpoint).lower() or "gpt2" in str(checkpoint).lower():
        raise ValueError("This diagnostic is scoped to the current Qwen checkpoint")
    response = tokenizer.encode("### Response:\n", add_special_tokens=False)
    if len(response) != 3:
        raise ValueError(f"Unexpected response-prefix tokens: {response}")
    if len(tokenizer) != config.vocab_size or tokenizer.eos_token_id != config.eos_token_id:
        raise ValueError("Tokenizer/config vocabulary or EOS mismatch")
    sids = sorted({line.split("\t")[0].strip() for line in catalog.read_text().splitlines() if line.strip()})
    prefix_ids = [tokenizer(f"### Response:\n{sid}\n").input_ids for sid in sids]
    callback = source_constraints(root, prefix_ids, tokenizer, len(response))
    expected = defaultdict(set)
    failures = []
    lengths = Counter()
    for sid, ids in zip(sids, prefix_ids):
        if ids[:3] != response:
            raise ValueError(f"Tokenized prefix differs for {sid}")
        suffix = ids[3:] + [tokenizer.eos_token_id]
        lengths[len(suffix)] += 1
        for depth, token in enumerate(suffix):
            key = tuple(response if depth == 0 else suffix[:depth])
            expected[(depth, key)].add(token)
            if token not in callback(0, list(key)):
                failures.append({"sid": sid, "depth": depth, "blocked_token": token})
    for (_, key), tokens in expected.items():
        if set(callback(0, list(key))) != tokens:
            raise ValueError(f"Unexpected allowed-token set for {key}")
    # Exercise the real stateful logits processor with 10 beams and two batches.
    # All paths above are audited; tensor-mask checks below sample up to 20 prefixes per step.
    processor = ConstrainedLogitsProcessor(callback, 10, str(checkpoint), config.eos_token_id)
    mask_checks = 0
    for depth in sorted({depth for depth, _ in expected}):
        keys = [key for d, key in expected if d == depth]
        selected = [keys[i * (len(keys) - 1) // (min(20, len(keys)) - 1)]
                    for i in range(min(20, len(keys)))] if len(keys) > 1 else keys
        selected = (selected + [selected[-1]] * 20)[:20]
        inputs = [list(response) if depth == 0 else response + list(key) for key in selected]
        if processor.count != depth:
            raise ValueError("Logits-processor step counter mismatch")
        scores = processor(torch.tensor(inputs), torch.zeros((20, len(tokenizer))))
        for i, key in enumerate(selected):
            finite = set(torch.isfinite(scores[i]).nonzero().flatten().tolist())
            if finite != expected[(depth, key)]:
                raise ValueError(f"Real logits mask differs at depth {depth}")
            mask_checks += 1
    return {"catalog_unique_sids": len(sids), "blocked_paths": len(failures),
            "blocked_examples": failures[:10], "suffix_lengths_including_eos": dict(lengths),
            "real_processor_mask_checks": mask_checks,
            "scope": "All catalog paths in actual evaluator hash builder; sampled real CPU masks with 10 beams. Does not test beam ranking/model probabilities.",
            "passed": not failures}


def compare_weights(base, checkpoint, torch):
    from safetensors import safe_open
    names = ["model.layers.0.self_attn.q_proj.weight", "model.layers.10.mlp.down_proj.weight",
             "model.layers.27.self_attn.o_proj.weight", "model.norm.weight"]
    result = []
    with safe_open(str(base / "model.safetensors"), framework="pt", device="cpu") as original, \
            safe_open(str(checkpoint / "model.safetensors"), framework="pt", device="cpu") as trained:
        for name in names:
            a, b = original.get_slice(name), trained.get_slice(name)
            if a.get_shape() != b.get_shape():
                raise ValueError(f"Shape mismatch: {name}")
            shape = a.get_shape()
            if len(shape) == 1:
                x, y = a[:], b[:]
                rows = "all"
            else:
                width = min(16, shape[0])
                rows = sorted({0, (shape[0] - width) // 2, shape[0] - width})
                x = torch.cat([a[start:start + width] for start in rows])
                y = torch.cat([b[start:start + width] for start in rows])
            dtype_before, dtype_after = str(x.dtype), str(y.dtype)
            x, y = x.float(), y.float()
            if not torch.isfinite(x).all() or not torch.isfinite(y).all():
                raise ValueError(f"Non-finite sampled weights: {name}")
            delta = y - x
            result.append({"name": name, "shape": shape, "sample_row_starts": rows,
                           "sample_elements": x.numel(), "base_dtype": dtype_before,
                           "sft_dtype": dtype_after,
                           "unchanged_fraction": (delta == 0).float().mean().item(),
                           "mean_absolute_delta": delta.abs().mean().item(),
                           "max_absolute_delta": delta.abs().max().item(),
                           "relative_rms_delta": (delta.square().mean().sqrt() /
                                                  x.square().mean().sqrt().clamp_min(1e-12)).item()})
    return {"tensors": result, "scope": "Selected backbone slices, not all parameters or new SID embeddings. Small/zero changes alone do not prove numerical rounding; requires interpretation."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, default=Path(__file__).resolve().parents[1] /
                        "data/Amazon/info/Industrial_and_Scientific_5_2016-10-2018-11.txt")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Choose a fresh diagnostic output: {args.output}")
    import torch
    torch.set_num_threads(4)
    root = Path(__file__).resolve().parents[1]
    result = {"base": str(args.base), "checkpoint": str(args.checkpoint), "torch_version": torch.__version__,
              "source_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                                for name in ("evaluate.py", "LogitProcessor.py", "sft.py", "scripts/diagnose_sft.py")}}
    for name, action in (("decoder", lambda: check_decoder(root, args.checkpoint, args.catalog, torch)),
                         ("weight_updates", lambda: compare_weights(args.base, args.checkpoint, torch))):
        try:
            result[name] = action()
        except Exception as error:
            result[name] = {"error": f"{type(error).__name__}: {error}"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
