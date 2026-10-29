"""Probe actual SFT SID/title prompts on catalog items; not held-out recommendation evaluation."""
import argparse
import json
from pathlib import Path
import random
import time
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from data import SidItemFeatDataset, Tokenizer

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--sample', type=int, default=5, help='Number of catalog rows per direction')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    if a.sample < 1:
        raise ValueError('sample must be positive')
    start = time.monotonic()
    tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    tok.pad_token_id = tok.eos_token_id
    # No tokenizer during construction: avoid eagerly encoding the entire catalog.
    # Attach the actual training wrapper and call pre() only for selected rows.
    ds = SidItemFeatDataset(
        item_file=str(ROOT/'data/Amazon/index/Industrial_and_Scientific.item.json'),
        index_file=str(ROOT/'data/Amazon/index/Industrial_and_Scientific.index.json'),
        tokenizer=None, max_len=512, test=True, seed=42)
    ds.tokenizer = Tokenizer(tok)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=torch.bfloat16, local_files_only=True).cuda().eval()
    catalog = set(ds.sid2title)
    records = []
    summary = {}
    normalize = lambda s: ' '.join(s.strip().split())
    for task in ('sid2title', 'title2sid'):
        eligible = [i for i, row in enumerate(ds.data) if row['task'] == task]
        chosen = random.Random(42).sample(eligible, min(a.sample, len(eligible)))
        for i in chosen:
            row = ds.data[i]
            encoded = ds.pre(i)
            ids = torch.tensor([encoded['input_ids']], device='cuda')
            mask = torch.tensor([encoded['attention_mask']], device='cuda')
            with torch.inference_mode():
                generated = model.generate(
                    input_ids=ids, attention_mask=mask, do_sample=False, num_beams=1,
                    max_new_tokens=128 if task == 'sid2title' else 16,
                    pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
            new_ids = generated[0, ids.shape[1]:].tolist()
            answer = tok.decode(new_ids, skip_special_tokens=True)
            record = {'task': task, 'catalog_dataset_row': i, 'input': row['input'],
                      'prompt': tok.decode(encoded['input_ids'], skip_special_tokens=False),
                      'reference': row['output'], 'prediction': answer,
                      'normalized_exact_match': normalize(answer)==normalize(row['output']),
                      'generated_tokens': len(new_ids), 'ended_with_eos': tok.eos_token_id in new_ids}
            if task == 'title2sid':
                record['in_catalog_sid'] = answer.strip('\n" ') in catalog
            records.append(record)
            print('\nTASK:', task, '\nINPUT:', row['input'], '\nREFERENCE:', row['output'],
                  '\nMODEL:', answer, flush=True)
        subset = [r for r in records if r['task']==task]
        summary[task] = {'queries': len(subset),
                         'normalized_exact_match': sum(r['normalized_exact_match'] for r in subset)/len(subset),
                         'no_eos_fraction': sum(not r['ended_with_eos'] for r in subset)/len(subset)}
        if task == 'title2sid':
            summary[task]['invalid_sid_fraction'] = sum(not r['in_catalog_sid'] for r in subset)/len(subset)
    result = {'model': a.model, 'seed': 42, 'generation': 'unconstrained greedy, no chat template; actual SidItemFeatDataset.pre(test=True)',
              'scope': 'Catalog alignment fitting probe. Catalog alignment rows were used in SFT; not held-out generalization or recommendation HR/NDCG. Existing dataset dictionaries collapse SID/title collisions; references follow actual training rows.',
              'summary': summary, 'examples': records, 'seconds': time.monotonic()-start}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(summary, indent=2), '\nSaved:', a.output)


if __name__ == '__main__':
    main()
