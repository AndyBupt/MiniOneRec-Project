"""Train-only tiny fitting/precision diagnostic; never replaces the E1 baseline."""

import argparse
import ast
import csv
import hashlib
import json
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def select_rows(rows, count, seed):
    """One representative per history, removing conflicting SID labels across full train."""
    if count < 1:
        raise ValueError("Sample count must be positive")
    targets, first = defaultdict(set), {}
    for index, row in enumerate(rows):
        history = tuple(ast.literal_eval(row["history_item_sid"]))
        targets[history].add(row["item_sid"].strip())
        first.setdefault(history, index)
    eligible = [first[h] for h, labels in targets.items() if len(labels) == 1 and h]
    if len(eligible) < count:
        raise ValueError(f"Need {count} eligible histories; found {len(eligible)}")
    indices = random.Random(seed).sample(eligible, count)
    return [rows[i] for i in indices], {
        "source_row_indices": indices, "train_rows": len(rows),
        "distinct_histories": len(targets),
        "conflicting_histories_excluded": sum(len(labels) > 1 for labels in targets.values()),
        "eligible_histories": len(eligible), "seed": seed,
    }


def sid_tokens(sid):
    match = re.fullmatch(r"(<a_\d+>)(<b_\d+>)(<c_\d+>)", sid)
    if not match:
        raise ValueError(f"Expected a three-layer SID: {sid}")
    return list(match.groups())


def validate_example(example, inference_ids, expected):
    ids, labels = example["input_ids"], example["labels"]
    supervised = [i for i, label in enumerate(labels) if label != -100]
    if (len(ids) != len(labels) or len(ids) != len(example["attention_mask"])
            or len(supervised) != 5 or [labels[i] for i in supervised] != expected
            or supervised != list(range(supervised[0], len(ids)))
            or any(labels[i] != ids[i] for i in supervised)
            or ids[:supervised[0]] != inference_ids):
        raise ValueError("Mask/target/truncation/prompt mismatch")


def prepare(args):
    folder = Path(args.output)
    if folder.exists():
        raise FileExistsError(f"Choose a fresh output directory: {folder}")
    with Path(args.train).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected, manifest = select_rows(rows, args.count, args.seed)
    folder.mkdir(parents=True)
    with (folder / "samples.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(selected)
    manifest.update({"source_train": str(Path(args.train).resolve()),
                     "source_sha256": hashlib.sha256(Path(args.train).read_bytes()).hexdigest(),
                     "sample_sha256": hashlib.sha256((folder / "samples.csv").read_bytes()).hexdigest(),
                     "samples": len(selected), "unique_target_sids": len({r["item_sid"] for r in selected}),
                     "purpose": "Train-only fitting diagnostic; easier non-conflicting subset; not held-out results"})
    write_json(folder / "samples_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


def fit(args):
    import sys
    import torch
    import torch.nn.functional as F
    import transformers
    from transformers import (AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq,
                              Trainer, TrainerCallback, TrainingArguments, set_seed)
    sys.path.insert(0, str(ROOT))
    from data import SidSFTDataset, EvalSidDataset

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("This diagnostic requires one BF16-capable CUDA GPU")
    if args.steps < 10 or args.micro_batch_size < 1 or args.batch_size < 1 or args.batch_size % args.micro_batch_size:
        raise ValueError("Need >=10 steps and batch_size divisible by micro_batch_size")
    out = Path(args.output)
    if out.exists():
        raise FileExistsError(f"Choose a fresh arm directory: {out}")
    checkpoint = Path(args.checkpoint).resolve()
    if out.resolve() == checkpoint or checkpoint in out.resolve().parents:
        raise ValueError("Diagnostic output must be outside the source checkpoint")
    out.mkdir(parents=True)
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    dataset = SidSFTDataset(args.samples, tokenizer, max_len=512, seed=args.seed,
                            category="industrial and scientific items")
    inference = EvalSidDataset(args.samples, tokenizer, max_len=512, seed=args.seed,
                               category="industrial and scientific items", test=True)
    with Path(args.samples).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(dataset) != len(rows) or len(inference) != len(rows):
        raise ValueError("Dataset row count changed")
    for index, (example, row) in enumerate(zip(dataset, rows)):
        expected = tokenizer.encode(row["item_sid"] + "\n", add_special_tokens=False) + [tokenizer.eos_token_id]
        validate_example(example, inference[index]["input_ids"], expected)
        sid_ids = [tokenizer.convert_tokens_to_ids(t) for t in sid_tokens(row["item_sid"])]
        if expected[:3] != sid_ids or len(set(sid_ids)) != 3:
            raise ValueError(f"SID is not three independent tokens at row {index}")
    write_json(out / "dataset_audit.json", {"samples": len(rows), "prompt_target_mask_checks_passed": True,
                                          "supervised_tokens_per_row": 5, "max_length": 512,
                                          "tasks": ["history_sid_to_next_sid"]})
    collator = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt")
    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=dtype, local_files_only=True)
    model.to("cuda")
    model.config.use_cache = False
    if len(tokenizer) != model.get_input_embeddings().weight.shape[0]:
        raise ValueError("Tokenizer/model vocabulary mismatch")
    if model.config.tie_word_embeddings and model.get_input_embeddings().weight.data_ptr() != model.get_output_embeddings().weight.data_ptr():
        raise ValueError("Expected shared input/output embeddings")
    if not all(p.requires_grad for p in model.parameters()):
        raise ValueError("Expected full-parameter diagnostic training")

    def teacher_metrics():
        model.eval()
        sums, total, layers = 0.0, 0, {name: Counter() for name in ("a", "b", "c")}
        with torch.no_grad():
            for start in range(0, len(dataset), args.micro_batch_size):
                batch = collator([dataset[i] for i in range(start, min(start + args.micro_batch_size, len(dataset)))])
                batch = {k: v.to("cuda") for k, v in batch.items()}
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
                for b in range(len(batch["labels"])):
                    positions = (batch["labels"][b] != -100).nonzero().flatten()
                    scores = logits[b, positions - 1].float()
                    targets = batch["labels"][b, positions]
                    losses = F.cross_entropy(scores, targets, reduction="none")
                    sums += losses.sum().item()
                    total += len(targets)
                    for j, name in enumerate(("a", "b", "c")):
                        state = layers[name]
                        state["nll"] += losses[j].item()
                        state["top1"] += int(scores[j].argmax() == targets[j])
                        state["top10"] += int((scores[j].topk(10).indices == targets[j]).any())
                del logits, batch
        return {"target_token_loss_including_newline_eos": sums / total,
                "sid_layers": {name: {k: v / len(dataset) for k, v in state.items()} for name, state in layers.items()},
                "scope": "Same training subset, unrestricted-vocab teacher forcing; gold earlier SID tokens; not generated accuracy"}

    names = ["model.layers.0.self_attn.q_proj.weight", "model.layers.10.mlp.down_proj.weight",
             "model.layers.27.self_attn.o_proj.weight", "model.norm.weight"]

    def snapshots():
        params = dict(model.named_parameters())
        result = {name: (params[name].detach() if params[name].ndim == 1 else params[name].detach()[:16])
                  .float().cpu().clone() for name in names}
        sid_rows = sorted({tokenizer.convert_tokens_to_ids(token) for row in rows
                           for token in sid_tokens(row["item_sid"])})
        result["target_sid_embedding_rows"] = model.get_input_embeddings().weight.detach()[sid_rows].float().cpu().clone()
        return result

    before_weights = snapshots()
    before = teacher_metrics()
    write_json(out / "teacher_before.json", before)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)

    class ResourceGate(TrainerCallback):
        def on_train_begin(self, args_, state, control, **kwargs):
            torch.cuda.synchronize()
            self.start = time.perf_counter()

        def on_step_end(self, args_, state, control, **kwargs):
            if state.global_step == 10:
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - self.start
                report = {"steps": 10, "seconds": elapsed,
                          "estimated_training_seconds": elapsed / 10 * args.steps,
                          "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                          "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                          "optimizer_state_dtypes": dict(Counter(str(v.dtype) for state_ in optimizer.state.values()
                                                                 for key, v in state_.items() if key != "step" and torch.is_tensor(v)))}
                write_json(out / "profile_step10.json", report)
                print("STEP10_PROFILE", json.dumps(report), flush=True)
                if report["peak_reserved_gib"] > args.max_memory_gib or report["estimated_training_seconds"] > args.max_seconds:
                    raise RuntimeError("Resource gate exceeded; profile saved. Stop and review before increasing limits.")

    training_args = TrainingArguments(
        output_dir=str(out / "trainer"), max_steps=args.steps,
        per_device_train_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=args.batch_size // args.micro_batch_size,
        learning_rate=args.lr, lr_scheduler_type="constant", warmup_steps=0,
        bf16=True, logging_steps=10, save_strategy="no", eval_strategy="no",
        report_to=[], seed=args.seed, data_seed=args.seed, max_grad_norm=1.0,
        dataloader_num_workers=0, dataloader_pin_memory=False,
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=dataset,
                      data_collator=collator, optimizers=(optimizer, None), callbacks=[ResourceGate()])
    set_seed(args.seed)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    result = trainer.train()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    resources = {"training_seconds": elapsed, "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                 "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30}
    after = teacher_metrics()
    updates = {}
    for name, value in snapshots().items():
        delta = value - before_weights[name]
        updates[name] = {"unchanged_fraction": (delta == 0).float().mean().item(),
                         "mean_absolute_delta": delta.abs().mean().item(),
                         "max_absolute_delta": delta.abs().max().item()}
    summary = {"precision": args.precision, "checkpoint": str(checkpoint), "samples": len(dataset),
               "sample_sha256": hashlib.sha256(Path(args.samples).read_bytes()).hexdigest(),
               "steps": args.steps, "lr": args.lr, "effective_batch_size": args.batch_size,
               "micro_batch_size": args.micro_batch_size, "seed": args.seed,
               "parameter_dtypes": dict(Counter(str(p.dtype) for p in model.parameters())),
               "torch": torch.__version__, "transformers": transformers.__version__,
               "gpu": torch.cuda.get_device_name(0), "teacher_before": before, "teacher_after": after,
               "sampled_updates_before_export": updates, "resources": resources, "trainer": result.metrics,
               "export": "Both arms exported BF16 for identical evaluate.py inference precision; optimizer states not saved",
               "status": "TRAINING_COMPLETE"}
    write_json(out / "summary.json", summary)
    model.config.use_cache = True
    model.to(dtype=torch.bfloat16)
    model.save_pretrained(out / "final_checkpoint", safe_serialization=True)
    tokenizer.save_pretrained(out / "final_checkpoint")
    (out / "COMPLETED").write_text("Training and BF16 export complete; see generated evaluation separately.\n")
    print(json.dumps(summary, indent=2))


def summarize(args):
    folder = Path(args.output)
    report = {"purpose": "Fixed train-only tiny-fit precision diagnosis, not generalization or E1 replacement", "arms": {}}
    for name in ("initial", "bf16", "fp32"):
        target = folder / name
        metrics = json.loads((target / "metrics.json").read_text())
        predictions = json.loads((target / "predictions.json").read_text())
        top1 = Counter(row["predict"][0].strip() for row in predictions)
        record = {"metrics": metrics,
                  "distinct_predicted_sids_at10": len({sid.strip() for row in predictions for sid in row["predict"][:10]}),
                  "distinct_top1_sids": len(top1), "most_common_top1": top1.most_common(1)}
        if name != "initial":
            record["training"] = json.loads((target / "summary.json").read_text())
        report["arms"][name] = record
    write_json(folder / "comparison.json", report)
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--train", required=True)
    prep.add_argument("--output", required=True)
    prep.add_argument("--count", type=int, default=128)
    prep.add_argument("--seed", type=int, default=42)
    arm = commands.add_parser("fit")
    for key in ("checkpoint", "samples", "output"):
        arm.add_argument(f"--{key}", required=True)
    arm.add_argument("--precision", choices=["bf16", "fp32"], required=True)
    arm.add_argument("--steps", type=int, default=100)
    arm.add_argument("--lr", type=float, default=2e-5)
    arm.add_argument("--seed", type=int, default=42)
    arm.add_argument("--batch-size", type=int, default=32)
    arm.add_argument("--micro-batch-size", type=int, default=2)
    arm.add_argument("--max-memory-gib", type=float, default=60)
    arm.add_argument("--max-seconds", type=float, default=1800)
    summary = commands.add_parser("summarize")
    summary.add_argument("--output", required=True)
    args = parser.parse_args()
    {"prepare": prepare, "fit": fit, "summarize": summarize}[args.command](args)


if __name__ == "__main__":
    main()
