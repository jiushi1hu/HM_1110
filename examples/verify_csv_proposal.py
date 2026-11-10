"""Evaluate frozen default policies under A/B/C action processing, without training.

A uses joint-feasible linear mapping. B/C use dated CSV proposal intervals with
physical correction; only C enables the existing valid-inverse correction cost.
Output/trace files are created only when explicitly requested, exclusively.
"""
import argparse
import copy
from contextlib import ExitStack
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.train_cascade import load_configuration
from harl.envs.cascade_reservoir.action_correction import action_correction_settings
from harl.envs.cascade_reservoir.correction_metrics import aggregate_action_correction_summaries
from harl.envs.cascade_reservoir.data_loader import CascadeData, IDS
from harl.envs.cascade_reservoir.episodes import episode_environment, positive_integer
from harl.envs.cascade_reservoir.periodic_evaluation import select_periodic_windows
from harl.envs.cascade_reservoir.reservoir_physics import InfeasibleCascade
from harl.envs.cascade_reservoir.reward import EpisodeRewardMetrics
from harl.runners.cascade_reservoir_runner import CascadeReservoirRunner, validate_seed
from harl.runners.cascade_mappo_runner import CascadeMAPPORunner

RUNNERS = dict(happo=CascadeReservoirRunner, mappo=CascadeMAPPORunner)
TRACE_FIELDS = ('algorithm', 'configuration', 'window_id', 'episode_step', 'date', 'reservoir_id',
    'sampled_action_index', 'normalized_action', 'realized_normalized_action',
    'storage_start_m3', 'storage_end_m3', 'inflow_m3s', 'external_inflow_m3s',
    'storage_lower_m3', 'storage_upper_m3', 'proposal_low_m3s', 'proposal_high_m3s',
    'physical_low_m3s', 'physical_high_m3s', 'target_release_m3s', 'executed_release_m3s',
    'release_correction_m3s', 'normalized_action_correction_magnitude',
    'scaled_release_correction', 'scaled_release_correction_valid', 'inverse_action_valid',
    'physical_controls', 'controls', 'forced_dispatch', 'degenerate_proposal',
    'correction_applied', 'numerical_adjustment', 'executed_outside_mapping_interval',
    'proposal_physical_disjoint', 'correction_reason', 'power_MW', 'generation_MWh',
    'ecological_flow_shortfall_m3s', 'guaranteed_output_shortfall_MW',
    'ecological_flow_violation', 'guaranteed_output_violation', 'local_correction_penalty')


def _mode_configs(physical):
    configs = {}
    for name, mode, penalty in (('A', 'joint_feasible', False),
                                ('B', 'csv_proposal', False), ('C', 'csv_proposal', True)):
        config = copy.deepcopy(physical)
        config['release_mapping_mode'] = mode
        config['action_correction'] = action_correction_settings(physical)
        config['action_correction'].update(enabled=name != 'A', penalty_enabled=penalty)
        configs[name] = config
    return configs


def _weights_digest(runner):
    digest = hashlib.sha256()
    for i, actor in enumerate(runner.actor):
        for name, value in sorted(actor.actor.state_dict().items()):
            digest.update(('actor'+str(i)+':'+name).encode())
            digest.update(value.detach().cpu().numpy().tobytes())
    for name, value in sorted(runner.critic.critic.state_dict().items()):
        digest.update(('critic:'+name).encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def _freeze(runner):
    runner.prep_rollout()
    for actor in runner.actor:
        for parameter in actor.actor.parameters():
            parameter.requires_grad_(False)
    for parameter in runner.critic.critic.parameters():
        parameter.requires_grad_(False)


def _commands(actors, env):
    observations = env.actor_observations()
    return np.asarray([int(actor.act(observations[i:i+1], deterministic=True).item())
                       for i, actor in enumerate(actors)], dtype=np.int64)


def _physical_check(row, env):
    release = row['executed_release']
    np.testing.assert_array_equal(release, row['release'])
    np.testing.assert_array_equal(row['inflow'][1:], row['external'][1:]+release[:-1])
    np.testing.assert_array_equal(row['storage_end'],
        row['storage_start']+(row['inflow']-release)*env.physics.dt)
    tolerance = np.maximum(1., 1e-10*np.abs(row['storage_start']))
    if ((row['storage_end'] < row['storage_lower_m3']-tolerance) |
            (row['storage_end'] > row['storage_upper_m3']+tolerance)).any():
        raise AssertionError('Executed storage violates the existing physical bounds')
    if ((release < -1e-8) | (release > env.data.release_cap+1e-8)).any():
        raise AssertionError('Executed release violates existing outlet capacity')


def _paired_check(b, c, commands_b, commands_c):
    np.testing.assert_array_equal(commands_b, commands_c)
    for key in ('storage_start', 'storage_end', 'release', 'target_release', 'power',
                'energy', 'inflow', 'controls', 'correction_applied', 'normalized_actions'):
        np.testing.assert_array_equal(b[key], c[key], err_msg='B/C physical mismatch: '+key)
    cost = c['local_action_correction_penalties']
    np.testing.assert_allclose(c['local_rewards'], b['local_rewards']-cost, rtol=0., atol=1e-12)
    np.testing.assert_allclose(c['mixed_rewards'], b['mixed_rewards']-.5*cost, rtol=0., atol=1e-12)
    np.testing.assert_allclose(c['system_reward'], b['system_reward']-.5*cost.sum(), rtol=0., atol=1e-12)
    if c['reward'] != b['reward'] or c['global_reward'] != b['global_reward']:
        raise AssertionError('Correction penalty changed legacy/global accounting')


def _trace_rows(algorithm, mode, window, step, row):
    direct = dict(storage_start_m3='storage_start', storage_end_m3='storage_end',
        inflow_m3s='inflow', external_inflow_m3s='external', storage_lower_m3='storage_lower_m3',
        storage_upper_m3='storage_upper_m3', proposal_low_m3s='proposal_low',
        proposal_high_m3s='proposal_high', physical_low_m3s='physical_low',
        physical_high_m3s='physical_high', target_release_m3s='target_release',
        executed_release_m3s='executed_release', release_correction_m3s='release_correction_m3s',
        scaled_release_correction='scaled_release_correction', power_MW='power',
        generation_MWh='energy', local_correction_penalty='local_action_correction_penalties')
    flags = ('scaled_release_correction_valid', 'inverse_action_valid', 'physical_controls',
        'controls', 'forced_dispatch', 'degenerate_proposal', 'correction_applied',
        'numerical_adjustment', 'executed_outside_mapping_interval', 'proposal_physical_disjoint')
    for i, reservoir in enumerate(IDS):
        record = dict(algorithm=algorithm, configuration=mode, window_id=window['window_id'],
            episode_step=step, date=row['date'], reservoir_id=reservoir,
            sampled_action_index=int(row['sampled_actions'][i]),
            normalized_action=float(row['normalized_actions'][i]),
            correction_reason=str(row['correction_reason'][i]))
        record.update({name:float(row[source][i]) for name,source in direct.items()})
        record.update({name:int(row[name][i]) for name in flags})
        valid = bool(row['inverse_action_valid'][i])
        record['realized_normalized_action'] = float(row['realized_normalized_actions'][i]) if valid else None
        record['normalized_action_correction_magnitude'] = float(row['normalized_action_correction_magnitude'][i]) if valid else None
        if not row['scaled_release_correction_valid'][i]:
            record['scaled_release_correction'] = None
        for term, field in (('ecological_flow', 'ecological_flow_shortfall_m3s'),
                            ('guaranteed_output', 'guaranteed_output_shortfall_MW')):
            shortfall = row.get('service_shortfalls', {}).get(term)
            record[field] = None if shortfall is None else float(shortfall[i])
            record[term+'_violation'] = None if shortfall is None else int(shortfall[i] > 0.)
        yield record


def _accumulator(env, gamma, physical):
    initial = env.storage.copy()
    return dict(metrics=EpisodeRewardMetrics(gamma, action_correction_config=physical),
        energy=np.zeros(5), storage_initial=initial, storage_min=initial.copy(),
        storage_max=initial.copy(), max_balance_residual_m3=0., max_system_residual_m3=0.,
        failure=None, completed_days=0, disjoint_examples=[])


def _collect(accumulator, row):
    accumulator['metrics'].update(row)
    accumulator['energy'] += row['energy']
    accumulator['storage_min'] = np.minimum(accumulator['storage_min'], row['storage_end'])
    accumulator['storage_max'] = np.maximum(accumulator['storage_max'], row['storage_end'])
    accumulator['max_balance_residual_m3'] = max(accumulator['max_balance_residual_m3'], float(np.abs(row['residual']).max()))
    accumulator['max_system_residual_m3'] = max(accumulator['max_system_residual_m3'], abs(row['system_residual']))
    accumulator['completed_days'] += 1
    for i in np.flatnonzero(row['proposal_physical_disjoint']):
        if len(accumulator['disjoint_examples']) >= 10:
            break
        accumulator['disjoint_examples'].append(dict(date=row['date'], reservoir_id=IDS[i],
            storage_start_m3=float(row['storage_start'][i]),
            cascade_storage_start_m3=row['storage_start'].tolist(),
            actual_inflow_m3s=float(row['inflow'][i]),
            proposal_low_m3s=float(row['proposal_low'][i]), proposal_high_m3s=float(row['proposal_high'][i]),
            physical_low_m3s=float(row['physical_low'][i]), physical_high_m3s=float(row['physical_high'][i]),
            target_release_m3s=float(row['target_release'][i]), executed_release_m3s=float(row['release'][i]),
            correction_reason=str(row['correction_reason'][i])))


def _window_result(accumulator, env, window):
    return dict(window=window, completed_days=accumulator['completed_days'],
        status='ok' if accumulator['completed_days'] == window['episode_horizon_days'] else 'incomplete',
        failure=accumulator['failure'],
        executed_performance=dict(generation_MWh_by_reservoir=accumulator['energy'].tolist(),
            generation_MWh=float(accumulator['energy'].sum()),
            service_requirements=accumulator['metrics'].shortfall_summary(),
            storage_initial_m3=accumulator['storage_initial'].tolist(), storage_terminal_m3=env.storage.tolist(),
            storage_minimum_m3=accumulator['storage_min'].tolist(), storage_maximum_m3=accumulator['storage_max'].tolist(),
            max_balance_residual_m3=accumulator['max_balance_residual_m3'],
            max_system_residual_m3=accumulator['max_system_residual_m3']),
        proposal_quality=accumulator['metrics'].action_correction_metrics.summary(),
        learning_reward_accounting=accumulator['metrics'].learning_summary(),
        disjoint_examples=accumulator['disjoint_examples'])


def _pooled_mode(windows):
    energy = sum((np.asarray(row['executed_performance']['generation_MWh_by_reservoir']) for row in windows), np.zeros(5))
    services = {}
    for term in ('guaranteed_output', 'ecological_flow'):
        counts = np.sum([row['executed_performance']['service_requirements'][term]['by_reservoir_days'] for row in windows], axis=0)
        active = np.sum([row['executed_performance']['service_requirements'][term]['by_reservoir_active_days'] for row in windows], axis=0)
        services[term] = dict(violation_count=int(counts.sum()), active_reservoir_days=int(active.sum()),
            violation_rate=float(counts.sum()/active.sum()) if active.sum() else None,
            by_reservoir={name:dict(violation_count=int(counts[i]), active_reservoir_days=int(active[i]),
                violation_rate=float(counts[i]/active[i]) if active[i] else None) for i,name in enumerate(IDS)})
    return dict(completed_windows=sum(row['status'] == 'ok' for row in windows),
        planned_windows=len(windows), executed_trajectory_days=sum(row['completed_days'] for row in windows),
        aggregation='pooled executed-day counts and summed energy; partial coverage explicitly labelled',
        executed_performance=dict(generation_MWh=float(energy.sum()), generation_TWh=float(energy.sum()/1e6),
            generation_MWh_by_reservoir=dict(zip(IDS, energy.tolist())), service_requirements=services),
        proposal_quality=aggregate_action_correction_summaries([row['proposal_quality'] for row in windows]),
        learning_reward_accounting=dict(
            discounted_system_returns_sum=sum(row['learning_reward_accounting']['episode_system_reward'] for row in windows),
            discounted_local_correction_cost_sum=sum(sum(row['learning_reward_accounting']['episode_local_action_correction_penalties']) for row in windows),
            discount_restarts_at_each_window=True),
        windows=windows)


def verify(days=365, seed=1, algorithms=('happo', 'mappo'), trace_writer=None):
    """Return full in-memory diagnostics; no training, checkpointing or file I/O."""
    days, seed = positive_integer(days, 'days'), validate_seed(seed)
    if not algorithms or len(set(algorithms)) != len(algorithms) or any(a not in RUNNERS for a in algorithms):
        raise ValueError('algorithms must be a unique nonempty happo/mappo selection')
    config, physical = load_configuration()
    physical['release_mapping_mode'] = 'csv_proposal'
    physical['action_correction'] = action_correction_settings(physical)
    physical['action_correction']['enabled'] = True
    data = CascadeData(ROOT/'data/cascade_reservoir_new', physical, config['experiment'])
    windows = select_periodic_windows(data, episode_days=days)
    modes = _mode_configs(physical)
    report = dict(schema='cascade_frozen_csv_verification_v1', seed=seed, episode_days=days,
        algorithms=list(algorithms), reservoir_ids=list(IDS), source_csv=data.release_bounds.metadata(),
        data_hashes=dict(data.hashes), physical_configurations=modes,
        policy_status='fresh default networks, frozen at initialization; no training or learned-performance claim',
        deterministic_policy=True, training_performed=False, optimizer_steps=0,
        model_configuration=copy.deepcopy(config['model']), ppo_configuration=copy.deepcopy(config['algo']),
        comparison='identical initial state and fixed weights across A/B/C; policies observe their own evolving environment state',
        penalty_scope='existing Eq.16 valid inside-proposal inverse only; scaled/outside corrections are descriptive',
        windows=windows, results={})
    for algorithm in algorithms:
        options = copy.deepcopy(config)
        options['train']['episode_length'] = days
        if algorithm == 'mappo':
            options['algo']['happo_credit_assignment'] = 'team_gae'
        runner = RUNNERS[algorithm](data, modes['A'], options, seed, periodic_windows=windows)
        try:
            _freeze(runner)
            initial_digest = _weights_digest(runner)
            collected = {mode:[] for mode in modes}
            paired_days = 0
            with torch.inference_mode():
                for window in windows:
                    envs = {mode:episode_environment(data, params, window['source_years'], days, runner.action_bins)
                            for mode,params in modes.items()}
                    initial = envs['A'].storage.copy()
                    for env in envs.values():
                        np.testing.assert_array_equal(env.storage, initial)
                    accumulators = {mode:_accumulator(env, config['algo']['gamma'], modes[mode]) for mode,env in envs.items()}
                    for day in range(1, days+1):
                        commands, rows, failed = {}, {}, False
                        for mode, env in envs.items():
                            commands[mode] = _commands(runner.actor, env)
                            try:
                                env.step(commands[mode])
                            except InfeasibleCascade as exc:
                                accumulators[mode]['metrics'].record_execution_infeasibility(getattr(exc, 'reservoir_index', None))
                                accumulators[mode]['failure'] = dict(error_type=type(exc).__name__, reason=str(exc),
                                    date=env.data.dates[min(env.index,env.stop-1)].isoformat(),
                                    diagnostics=getattr(exc, 'diagnostics', None))
                                failed = True
                                continue
                            row = env.last_transition
                            _physical_check(row, env)
                            _collect(accumulators[mode], row)
                            rows[mode] = row
                            if trace_writer is not None:
                                trace_writer.writerows(_trace_rows(algorithm, mode, window, day, row))
                        if 'B' in rows and 'C' in rows:
                            _paired_check(rows['B'], rows['C'], commands['B'], commands['C'])
                            paired_days += 1
                        if failed:
                            break
                    for mode, env in envs.items():
                        collected[mode].append(_window_result(accumulators[mode], env, window))
            final_digest = _weights_digest(runner)
            if initial_digest != final_digest or runner.training_steps != 0 or runner.update_count != 0:
                raise AssertionError('Frozen verification changed a network or performed an update')
            if any(len(actor.actor_optimizer.state) for actor in runner.actor) or len(runner.critic.critic_optimizer.state):
                raise AssertionError('Verification advanced optimizer state')
            report['results'][algorithm] = dict(policy_weights_sha256=initial_digest,
                weights_unchanged=True, optimizer_steps=0, paired_B_C_days=paired_days,
                configurations={mode:_pooled_mode(rows) for mode,rows in collected.items()})
        finally:
            runner.close()
    return report


def compact_report(report):
    """Retain full source provenance while avoiding repeated full per-day schemas."""
    result = {key:report[key] for key in ('schema', 'seed', 'episode_days', 'algorithms', 'source_csv',
        'policy_status', 'training_performed', 'optimizer_steps', 'comparison', 'penalty_scope')}
    result['results'] = {}
    for algorithm, row in report['results'].items():
        modes = {}
        for mode, value in row['configurations'].items():
            audit = value['proposal_quality']
            modes[mode] = dict(completed_windows=value['completed_windows'], planned_windows=value['planned_windows'],
                executed_trajectory_days=value['executed_trajectory_days'],
                generation_TWh=value['executed_performance']['generation_TWh'],
                service_violation_rates={term:info['violation_rate'] for term,info in value['executed_performance']['service_requirements'].items()},
                proposal_quality={key:audit[key] for key in ('observed_reservoir_days', 'physical_nonforced_reservoir_days',
                    'physical_intervention_count', 'physical_intervention_rate', 'valid_controllable_reservoir_days',
                    'correction_violation_count', 'correction_violation_rate', 'outside_correction_events',
                    'forced_dispatch_events', 'degenerate_proposal_events', 'proposal_physical_disjoint_events',
                    'scaled_release_correction_mean', 'discharge_correction_abs_mean_m3s')},
                learning_reward_accounting=value['learning_reward_accounting'],
                storage_by_window=[dict(window_id=w['window']['window_id'], status=w['status'],
                    initial_m3=w['executed_performance']['storage_initial_m3'], terminal_m3=w['executed_performance']['storage_terminal_m3'],
                    minimum_m3=w['executed_performance']['storage_minimum_m3'], maximum_m3=w['executed_performance']['storage_maximum_m3']) for w in value['windows']],
                first_disjoint_examples=[dict(example, window_id=w['window']['window_id'])
                    for w in value['windows'] for example in w['disjoint_examples']][:10])
        result['results'][algorithm] = dict(weights_unchanged=row['weights_unchanged'],
            policy_weights_sha256=row['policy_weights_sha256'], paired_B_C_days=row['paired_B_C_days'], configurations=modes)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--days', type=int, default=365)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--algorithms', nargs='+', choices=tuple(RUNNERS), default=list(RUNNERS))
    parser.add_argument('--output', type=Path, help='Optional new full JSON report; existing files are never overwritten')
    parser.add_argument('--trace', type=Path, help='Optional new daily reservoir CSV; existing files are never overwritten')
    args = parser.parse_args(argv)
    paths = [path for path in (args.output, args.trace) if path is not None]
    if len({path.resolve() for path in paths}) != len(paths):
        parser.error('Output and trace paths must be distinct')
    if any(path.exists() for path in paths):
        parser.error('Output and trace paths must not already exist')
    try:
        with ExitStack() as stack:
            writer = None
            if args.trace is not None:
                stream = stack.enter_context(args.trace.open('x', encoding='utf-8', newline=''))
                writer = csv.DictWriter(stream, fieldnames=TRACE_FIELDS)
                writer.writeheader()
            report = verify(args.days, args.seed, args.algorithms, writer)
            if args.output is not None:
                with args.output.open('x', encoding='utf-8') as stream:
                    json.dump(report, stream, indent=2, allow_nan=False)
                    stream.write('\n')
            print(json.dumps(compact_report(report), indent=2, allow_nan=False))
    except (ValueError, FileExistsError) as exc:
        parser.error(str(exc))
    return 0 if all(value['completed_windows'] == value['planned_windows']
                    for row in report['results'].values() for value in row['configurations'].values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
