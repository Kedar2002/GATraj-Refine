"""Evaluate all modes, density groups, label-free top-1 collisions and latency.

Use the same arguments as train.py, adding --load_model <epoch>.
No parameters are selected or updated on the held-out dataset.
"""
import json
import time
from pathlib import Path

import numpy as np
import torch

from Processor import Processor
from train import parse_run_args


def collision_counts(prediction, target, splits, radius=0.2):
    """Unordered agent-window pairs closer than radius at any predicted step."""
    predicted, actual, pairs = 0, 0, 0
    for left, right in splits:
        left, right = int(left), int(right)
        n = right - left
        if n < 2:
            continue
        indices = torch.triu_indices(n, n, offset=1, device=prediction.device)
        for trajectories, is_prediction in ((prediction, True), (target, False)):
            positions = trajectories[left:right].transpose(0, 1)
            separation = torch.cdist(positions, positions).amin(dim=0)
            count = (separation[indices[0], indices[1]] < radius).sum().item()
            if is_prediction:
                predicted += count
            else:
                actual += count
        pairs += indices.shape[1]
    return predicted, actual, pairs


@torch.no_grad()
def evaluate(processor):
    args = processor.args
    model = processor.net.eval()
    device = next(model.parameters()).device
    groups = {name: {'agents': 0, 'ade_sum': 0., 'fde_sum': 0.}
              for name in ('all', 'sparse', 'crowded')}
    forward_times, timed_agents = [], 0
    predicted_collisions, actual_collisions, pair_count = 0, 0, 0
    loader = processor.dataloader_gt
    for batch in range(loader.testbatchnums):
        values, splits, neighbors = loader.get_test_batch(batch, args.load_model)
        absolute, normalized, shift, sequence_mask, counts = (
            torch.as_tensor(value, dtype=torch.float32, device=device) for value in values)
        inputs = absolute, normalized, neighbors, counts, splits
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        means, scales, logits = model(inputs, args.load_model, iftest=True, return_mixture=True)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        if batch >= 3:  # Exclude the first three batches as warmup.
            forward_times.append(elapsed)
            timed_agents += means.shape[1]

        target = normalized[args.obs_length:, :, :2].transpose(0, 1)
        errors = (means - target[None]).norm(dim=-1)
        ade, fde = errors.mean(dim=-1).amin(dim=0), errors[:, :, -1].amin(dim=0)
        degree = torch.zeros_like(ade)
        for (left, right), neighbor in zip(splits, neighbors):
            # Use the original observed graph for the same density groups in all ablations.
            base_mask = np.asarray(neighbor[args.obs_length - 1]) > 0
            degree[int(left):int(right)] = torch.as_tensor(base_mask.sum(1), device=device)
        for name, select in (('all', torch.ones_like(degree, dtype=torch.bool)),
                             ('sparse', degree <= 2), ('crowded', degree > 2)):
            groups[name]['agents'] += select.sum().item()
            groups[name]['ade_sum'] += ade[select].sum().item()
            groups[name]['fde_sum'] += fde[select].sum().item()

        # Each agent's most probable mode; no oracle labels and no assumption
        # that mode k has shared meaning across different agents.
        mode = logits.argmax(dim=-1)
        chosen = means[mode, torch.arange(means.shape[1], device=device)]
        origin = absolute[args.obs_length - 1, :, None, :2]
        predicted, actual, pairs = collision_counts(chosen + origin, target + origin, splits)
        predicted_collisions += predicted
        actual_collisions += actual
        pair_count += pairs

    for group in groups.values():
        count = group['agents']
        group['minADE'] = group.pop('ade_sum') / count if count else None
        group['minFDE'] = group.pop('fde_sum') / count if count else None
    report = {
        'test_set': args.test_set, 'seed': args.seed, 'epoch': args.load_model,
        'split': 'validation' if args.validation_set >= 0 else 'test',
        'evaluation_scene': args.validation_set if args.validation_set >= 0 else args.test_set,
        'variant': ('refined' if args.refiner else 'baseline') if args.stage == 'refiner' else 'backbone',
        'base_sha256': args.base_sha256,
        'base_epoch': getattr(model, 'base_epoch', None),
        'num_modes': args.final_mode,
        'groups': groups,
        'parameters': sum(p.numel() for p in model.parameters()),
        'active_parameters': sum(p.numel() for p in (model.backbone if args.stage == 'refiner' and not args.refiner else model).parameters()),
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'device': torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU',
        'torch_version': str(torch.__version__),
        'forward_ms_per_batch': 1000 * sum(forward_times) / len(forward_times) if forward_times else None,
        'forward_microseconds_per_agent': 1e6 * sum(forward_times) / timed_agents if timed_agents else None,
        'timed_batches': len(forward_times), 'warmup_batches': 3,
        'collision_diagnostic': {
            'radius_metres': 0.2, 'agent_window_pairs': pair_count,
            'top1_predicted_collision_pairs': predicted_collisions,
            'ground_truth_collision_pairs': actual_collisions,
            'top1_predicted_pair_rate': predicted_collisions / pair_count if pair_count else None,
            'ground_truth_pair_rate': actual_collisions / pair_count if pair_count else None,
        },
    }
    destination = Path(args.model_dir) / 'evaluation_{}_{}.json'.format(args.load_model, report['variant'])
    destination.write_text(json.dumps({**report, 'experiment': vars(args)}, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    print('Saved:', destination)
    return report


if __name__ == '__main__':
    arguments = parse_run_args(default_phase='test')
    if arguments.phase != 'test' or arguments.load_model <= 0:
        raise ValueError('Evaluation requires --phase test and --load_model <epoch>')
    evaluate(Processor(arguments))
