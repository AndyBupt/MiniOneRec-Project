"""Score saved validation candidates against frozen E1/E2; no generation or training."""
import argparse
from contextlib import nullcontext
import json
import math
from pathlib import Path
import random
import statistics
import time


def summarize(values):
    finite = sorted(x for x in values if math.isfinite(x))
    return {"count": len(values), "nonfinite": len(values) - len(finite),
            "median": statistics.median(finite) if finite else None,
            "p95": finite[int(.95 * (len(finite)-1))] if finite else None,
            "max": max(finite) if finite else None}


def main():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', required=True)
    p.add_argument('--policy', required=True)
    p.add_argument('--predictions', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--sample', type=int, default=128)
    p.add_argument('--batch-size', type=int, default=8)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    if a.sample < 1 or a.batch_size < 1:
        raise ValueError('Positive sample and batch size required')
    start = time.monotonic()
    tok = AutoTokenizer.from_pretrained(a.policy, local_files_only=True, padding_side='left')
    ref_tok = AutoTokenizer.from_pretrained(a.reference, local_files_only=True)
    if tok.get_vocab() != ref_tok.get_vocab():
        raise ValueError('Policy/reference vocabularies differ')
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    records = json.loads(a.predictions.read_text())
    indices = sorted(random.Random(42).sample(range(len(records)), min(a.sample, len(records))))
    rows = []
    for i in indices:
        r = records[i]
        for rank, sid in enumerate(r['predict'][:10], 1):
            completion = tok.encode(sid.strip() + '\n', add_special_tokens=False) + [tok.eos_token_id]
            rows.append((r['source_row_index'], r['input'], sid.strip(), rank, completion))
    # Mirror the original entry point's SDPA settings.
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)

    def score(model_path, bf16):
        print('Loading', model_path, 'autocast BF16:', bf16, flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float32, local_files_only=True).cuda().eval()
        results = []
        with torch.inference_mode():
            for offset in range(0, len(rows), a.batch_size):
                batch = rows[offset:offset+a.batch_size]
                prompts = [tok.encode(x[1], add_special_tokens=False)[-512:] for x in batch]
                width = max(map(len, prompts))
                length = max(len(x[4]) for x in batch)
                ids, masks = [], []
                for prompt, row in zip(prompts, batch):
                    comp = row[4]
                    ids.append([tok.pad_token_id]*(width-len(prompt)) + prompt + comp + [tok.pad_token_id]*(length-len(comp)))
                    masks.append([0]*(width-len(prompt)) + [1]*(len(prompt)+len(comp)) + [0]*(length-len(comp)))
                ids = torch.tensor(ids, device='cuda')
                mask = torch.tensor(masks, device='cuda')
                context = torch.autocast('cuda', dtype=torch.bfloat16) if bf16 else nullcontext()
                with context:
                    logits = model(input_ids=ids, attention_mask=mask,
                                   logits_to_keep=length+1, use_cache=False).logits[:, :-1]
                    # Float log-softmax provides a comparison with explicit stable arithmetic.
                    selected = logits.float().log_softmax(-1).gather(-1, ids[:, -length:, None]).squeeze(-1)
                results.extend(selected[i, :len(row[4])].cpu().tolist() for i, row in enumerate(batch))
                if offset % (a.batch_size*20) == 0:
                    print('Scored', min(offset+len(batch), len(rows)), '/', len(rows), flush=True)
        del model
        torch.cuda.empty_cache()
        return results

    ref = score(a.reference, False)
    policy_fp32 = score(a.policy, False)
    policy_bf16 = score(a.policy, True)
    details = []
    for row, rr, pp, bb in zip(rows, ref, policy_fp32, policy_bf16):
        for j, (r, f, b) in enumerate(zip(rr, pp, bb)):
            def kl(delta):
                return math.expm1(delta)-delta if delta < 709 else float('inf')
            details.append({'source_row_index': row[0], 'rank': row[3], 'sid': row[2],
                            'token_position': j, 'token_id': row[4][j],
                            'token': tok.convert_ids_to_tokens(row[4][j]),
                            'reference_logp_fp32': r, 'policy_logp_fp32': f, 'policy_logp_bf16': b,
                            'kl_fp32': kl(r-f), 'kl_mixed': kl(r-b), 'precision_logp_gap': abs(f-b)})
    output = {'scope': 'Frozen final-checkpoint saved-valid-candidate diagnostic. Not replay of training step 2362; eval mode, no optimizer, no gradients. Explicit FP32 log-softmax; not bitwise replay of selective_log_softmax.',
              'reference': a.reference, 'policy': a.policy, 'sample_queries': len(indices),
              'completion_sequences': len(rows), 'completion_lengths': sorted(set(len(r[4]) for r in rows)),
              'summary': {k: summarize([r[k] for r in details]) for k in ['kl_fp32', 'kl_mixed', 'precision_logp_gap']},
              'worst_kl_tokens': sorted(details, key=lambda x:x['kl_mixed'], reverse=True)[:30],
              'seconds': time.monotonic()-start}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(output, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(output['summary'], indent=2), '\nSaved:', a.output)


if __name__ == '__main__':
    main()
