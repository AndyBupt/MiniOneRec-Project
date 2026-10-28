"""Frozen E1/E2 padding and generation-position audit; no training or weight changes."""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import random
import time


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', required=True)
    p.add_argument('--policy', required=True)
    p.add_argument('--predictions', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--sample', type=int, default=12)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    if a.sample < 1:
        raise ValueError('sample must be positive')
    start = time.monotonic()
    tok = AutoTokenizer.from_pretrained(a.policy, local_files_only=True)
    rt = AutoTokenizer.from_pretrained(a.reference, local_files_only=True)
    if tok.get_vocab() != rt.get_vocab():
        raise ValueError('Reference/policy vocabularies differ')
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    records = json.loads(a.predictions.read_text())
    # Include the previously observed extreme row, plus fixed random queries.
    extreme = next((i for i, x in enumerate(records) if x['source_row_index'] == 4398), None)
    indices = random.Random(42).sample(range(len(records)), min(a.sample, len(records)))
    if extreme is not None and extreme not in indices:
        indices[0] = extreme
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    details = []
    for label, path in [('reference', a.reference), ('policy', a.policy)]:
        print('Loading', label, flush=True)
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32,
                                                    local_files_only=True).cuda().eval()
        for bf16 in (False, True):
            for i in indices:
                r = records[i]
                prompt = tok.encode(r['input'], add_special_tokens=False)[-512:]
                comp = tok.encode(r['predict'][0].strip()+'\n', add_special_tokens=False)+[tok.eos_token_id]
                seq = prompt + comp
                context = lambda: torch.autocast('cuda', dtype=torch.bfloat16) if bf16 else nullcontext()

                def forward(padding, explicit):
                    ids = torch.tensor([[pad]*padding+seq], device='cuda')
                    mask = torch.tensor([[0]*padding+[1]*len(seq)], device='cuda')
                    kwargs = {}
                    if explicit:
                        pos = mask.long().cumsum(-1)-1
                        pos.masked_fill_(mask == 0, 1)
                        kwargs['position_ids'] = pos
                    with torch.inference_mode(), context():
                        logits = model(input_ids=ids, attention_mask=mask, use_cache=False,
                                       logits_to_keep=len(comp)+1, **kwargs).logits[:, :-1]
                        target = torch.tensor(comp, device='cuda').view(1, -1, 1)
                        return logits.float().log_softmax(-1).gather(-1, target).flatten().cpu()

                single = forward(0, False)
                explicit_single = forward(0, True)
                for padding in (17, 97):
                    implicit = forward(padding, False)
                    explicit = forward(padding, True)
                    # Actual generate first-step raw logits, with the same padded prompt.
                    ids = torch.tensor([[pad]*padding+prompt], device='cuda')
                    mask = torch.tensor([[0]*padding+[1]*len(prompt)], device='cuda')
                    cfg = GenerationConfig(max_new_tokens=1, do_sample=False, num_beams=1,
                                           pad_token_id=pad, eos_token_id=tok.eos_token_id,
                                           return_dict_in_generate=True, output_logits=True,
                                           use_cache=True)
                    with torch.inference_mode(), context():
                        generated = model.generate(input_ids=ids, attention_mask=mask,
                                                   generation_config=cfg)
                        gen_logp = generated.logits[0].float().log_softmax(-1)[0, comp[0]].item()
                    details.append({'model': label, 'bf16_autocast': bf16,
                                    'source_row_index': r['source_row_index'], 'left_padding': padding,
                                    'prompt_tokens': len(prompt),
                                    'single_logps': single.tolist(), 'implicit_pad_logps': implicit.tolist(),
                                    'explicit_pad_logps': explicit.tolist(),
                                    'single_explicit_max_gap': (single-explicit_single).abs().max().item(),
                                    'implicit_padding_max_gap': (single-implicit).abs().max().item(),
                                    'explicit_padding_max_gap': (single-explicit).abs().max().item(),
                                    'generate_first_token_logp': gen_logp,
                                    'generate_vs_single_first_gap': abs(gen_logp-single[0].item()),
                                    'generate_vs_implicit_first_gap': abs(gen_logp-implicit[0].item()),
                                    'generate_vs_explicit_first_gap': abs(gen_logp-explicit[0].item())})
            print(label, 'BF16:', bf16, 'completed', len(indices), 'queries', flush=True)
        del model
        torch.cuda.empty_cache()
    keys = ['single_explicit_max_gap', 'implicit_padding_max_gap', 'explicit_padding_max_gap',
            'generate_vs_single_first_gap', 'generate_vs_implicit_first_gap', 'generate_vs_explicit_first_gap']
    summary = {}
    for label in ('reference', 'policy'):
        for bf16 in (False, True):
            subset = [r for r in details if r['model']==label and r['bf16_autocast']==bf16]
            summary[f'{label}_'+('bf16' if bf16 else 'fp32')] = {
                k: max(r[k] for r in subset) for k in keys}
    result = {'scope': 'Frozen eval-mode artificial left-padding audit. Actual generate raw first-step logits compared to forward. No constraint decoding, training, optimizer, or replay of historical spike. Numerical gaps require interpretation, not automatic proof of a bug.',
              'reference': a.reference, 'policy': a.policy, 'sample_queries': len(indices),
              'summary_max_logp_gaps': summary, 'details': details,
              'seconds': time.monotonic()-start}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(summary, indent=2), '\nSaved:', a.output)


if __name__ == '__main__':
    main()
