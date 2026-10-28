"""Bounded local KL event records; detached tensors only, no objective changes."""
import json
import math
from pathlib import Path


def record_kl_event(trainer, inputs, policy_logps, reference_logps, token_kl, threshold=10000., limit=32):
    mask = inputs['completion_mask'].bool()
    values = token_kl.detach().masked_select(mask)
    if not values.numel():
        return
    maximum = values.max().item()
    trainer._metrics['kl_token_max'].append(maximum)
    if math.isfinite(maximum) and maximum < threshold:
        return
    count = getattr(trainer, '_kl_event_count', 0)
    if count >= limit:
        return
    trainer._kl_event_count = count + 1
    def safe(value):
        value = float(value)
        return value if math.isfinite(value) else str(value)
    ranked = token_kl.detach().masked_fill(~mask, float('-inf')).flatten()
    top = ranked.topk(min(10, int(mask.sum().item()))).indices.tolist()
    width = token_kl.shape[1]
    tokens = []
    for index in top:
        row, column = divmod(index, width)
        tokens.append({'row': row, 'position': column,
                       'token_id': int(inputs['completion_ids'][row, column].item()),
                       'sample_id': inputs.get('diagnostic_sample_ids', [None]*token_kl.shape[0])[row],
                       'task_type': inputs.get('diagnostic_task_types', [None]*token_kl.shape[0])[row],
                       'reference_logp': safe(reference_logps[row, column].detach().item()),
                       'policy_logp': safe(policy_logps[row, column].detach().item()),
                       'kl': safe(token_kl[row, column].detach().item()),
                       'completion_ids': inputs['completion_ids'][row].detach().cpu().tolist()})
    path = Path(trainer.args.output_dir) / f'kl_events_rank{trainer.accelerator.process_index}.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {'step': trainer.state.global_step, 'training': trainer.model.training,
               'policy_logp_dtype': str(policy_logps.dtype), 'reference_logp_dtype': str(reference_logps.dtype),
               'maximum_token_kl': safe(maximum), 'event_number': count+1, 'tokens': tokens}
    with path.open('a') as handle:
        handle.write(json.dumps(payload, allow_nan=False)+'\n')
