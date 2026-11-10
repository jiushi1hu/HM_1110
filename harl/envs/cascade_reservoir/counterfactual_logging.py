"""Exclusive, streamed evidence for frozen model-assisted actor credit."""
import csv
import gzip
from pathlib import Path
import numpy as np
from harl.envs.cascade_reservoir.data_loader import IDS
from harl.envs.cascade_reservoir.batched_physics import batched_transition

COUNTERFACTUAL_TRACE_FIELDS = (
    'seed', 'update', 'episode', 'parallel_env_index', 'episode_step', 'date',
    'reservoir_id', 'controllable', 'sampled_action_index', 'joint_action_indices',
    'system_reward', 'team_gae_advantage', 'q_selected', 'counterfactual_baseline',
    'counterfactual_raw_advantage', 'counterfactual_training_advantage', 'advantage_scale',
    'best_action_index', 'q_best', 'best_q_delta', 'selected_action_probability',
    'best_action_probability', 'old_policy_probabilities', 'frozen_q_values',
    'normalized_action', 'target_release_m3s', 'executed_release_m3s',
    'release_correction_m3s', 'realized_normalized_action', 'inverse_action_valid',
    'normalized_action_correction_magnitude', 'correction_deficit', 'correction_applied',
    'correction_reason', 'forced_dispatch', 'numerical_adjustment',
    'executed_outside_mapping_interval',
    'proposal_low_m3s','proposal_high_m3s','physical_low_m3s','physical_high_m3s',
    'physical_controllable','degenerate_proposal','scaled_release_correction',
    'scaled_release_correction_valid','proposal_physical_disjoint',
)


class CounterfactualCreditWriter:
    """Retain candidate values and the exact actor credit before optimization."""
    def __init__(self, path):
        self.path = Path(path)
        self.stream = self.writer = None
        self.rows_written = 0
        self.last_update = 0

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = gzip.open(self.path, 'xt', encoding='utf-8', newline='')
        self.writer = csv.DictWriter(self.stream, fieldnames=COUNTERFACTUAL_TRACE_FIELDS,
                                     lineterminator='\n')
        self.writer.writeheader()
        self.stream.flush()
        return self

    def __exit__(self, *unused):
        if self.stream is not None:
            self.stream.close()
        self.stream = self.writer = None

    def write_batch(self, runner):
        if self.stream is None or self.stream.closed:
            raise RuntimeError('Counterfactual trace writer must be used in its context manager')
        update = runner.update_count+1
        if update <= self.last_update:
            raise ValueError('Counterfactual trace updates must increase')
        credit, context = runner.counterfactual_credit, runner.counterfactual_context
        probability = runner.counterfactual_probabilities
        cb = runner.critic_buffer
        # Reconstruct selected-command audit fields from the same frozen physical
        # context as the Q grid; no environment state, policy, or RNG is advanced.
        audit = batched_transition(runner.envs[0].physics,
            context['storage_start'].reshape(-1, 5), context['external'].reshape(-1, 5),
            context['actions'].reshape(-1, 5)/(probability.shape[-1]-1),
            context['storage_lower'].reshape(-1, 5), context['storage_upper'].reshape(-1, 5),
            **{key:context[key].reshape(-1,5) for key in ('proposal_low','proposal_high') if key in context})
        audit = {key:value.reshape(runner.length, runner.n_envs, 5)
                 for key,value in audit.items() if value.shape == (runner.length*runner.n_envs, 5)}
        prediction = cb.value_preds[:-1]
        if runner.value_normalizer is not None:
            prediction = runner.value_normalizer.denormalize(prediction)
        team_advantage = cb.returns[:-1,...,0]-prediction[...,0]
        for t in range(runner.length):
            for lane in range(runner.n_envs):
                for i, reservoir in enumerate(IDS):
                    selected = int(context['actions'][t,lane,i])
                    best = int(credit['best_action_indices'][t,lane,i])
                    q = credit['q_values'][t,lane,i]
                    p = probability[t,lane,i]
                    row = dict(seed=runner.seed, update=update,
                        episode=context['episode_ids'][lane], parallel_env_index=lane,
                        episode_step=t+1, date=context['dates'][t,lane], reservoir_id=reservoir,
                        controllable=int(context['controls'][t,lane,i] > 0),
                        sampled_action_index=selected,
                        joint_action_indices='['+','.join(map(str,context['actions'][t,lane]))+']',
                        system_reward=float(cb.rewards[t,lane,0]),
                        team_gae_advantage=float(team_advantage[t,lane]),
                        q_selected=float(credit['q_actual'][t,lane,i]),
                        counterfactual_baseline=float(credit['baseline'][t,lane,i]),
                        counterfactual_raw_advantage=float(credit['raw_advantages'][t,lane,i]),
                        counterfactual_training_advantage=float(runner.counterfactual_advantages[t,lane,i]),
                        advantage_scale=float(credit['advantage_scale']),
                        best_action_index=best, q_best=float(q[best]),
                        best_q_delta=float(q[best]-q[selected]),
                        selected_action_probability=float(p[selected]),
                        best_action_probability=float(p[best]),
                        old_policy_probabilities='['+','.join(format(float(x),'.8g') for x in p)+']',
                        frozen_q_values='['+','.join(format(float(x),'.12g') for x in q)+']')
                    valid = bool(audit['inverse_action_valid'][t,lane,i])
                    row.update(normalized_action=float(audit['normalized_actions'][t,lane,i]),
                        target_release_m3s=float(audit['target_release'][t,lane,i]),
                        executed_release_m3s=float(audit['executed_release'][t,lane,i]),
                        release_correction_m3s=float(audit['release_correction_m3s'][t,lane,i]),
                        realized_normalized_action=float(audit['realized_normalized_actions'][t,lane,i]) if valid else None,
                        inverse_action_valid=int(valid),
                        normalized_action_correction_magnitude=float(audit['normalized_action_correction_magnitude'][t,lane,i]) if valid else None,
                        correction_deficit=float(audit['correction_deficit'][t,lane,i]) if valid else None,
                        correction_applied=int(audit['correction_applied'][t,lane,i]),
                        correction_reason=str(audit['correction_reason'][t,lane,i]),
                        forced_dispatch=int(audit['forced_dispatch'][t,lane,i]),
                        numerical_adjustment=int(audit['numerical_adjustment'][t,lane,i]),
                        executed_outside_mapping_interval=int(audit['executed_outside_mapping_interval'][t,lane,i]))
                    for column, source in {
                        'proposal_low_m3s':'proposal_low','proposal_high_m3s':'proposal_high',
                        'physical_low_m3s':'physical_low','physical_high_m3s':'physical_high',
                        'physical_controllable':'physical_controls','degenerate_proposal':'degenerate_proposal',
                        'scaled_release_correction':'scaled_release_correction',
                        'scaled_release_correction_valid':'scaled_release_correction_valid',
                        'proposal_physical_disjoint':'proposal_physical_disjoint'}.items():
                        row[column] = float(audit[source][t,lane,i])
                    if not audit['scaled_release_correction_valid'][t,lane,i]:
                        row['scaled_release_correction'] = None
                    if any(isinstance(value,float) and not np.isfinite(value) for value in row.values()):
                        raise FloatingPointError('Non-finite counterfactual trace value')
                    self.writer.writerow(row)
                    self.rows_written += 1
                self.stream.flush()
        self.last_update = update
