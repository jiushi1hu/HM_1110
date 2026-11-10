"""Process-parallel cascade physics with explicit reset/step/close ownership."""
import multiprocessing as mp
import os
import traceback
import numpy as np
from harl.envs.cascade_reservoir.episodes import episode_environment, positive_integer
from harl.envs.cascade_reservoir.reward import EpisodeRewardMetrics
from harl.envs.cascade_reservoir.training_metrics import episode_record
from harl.envs.cascade_reservoir.reservoir_physics import InfeasibleCascade


def _worker(connection):
    env = metrics = energy = None
    episode = 0
    try:
        # A small startup message avoids blocking Process.start() while it
        # serializes the full dataset to a child that failed during imports.
        connection.send((True, dict(ready=True, pid=os.getpid())))
        command, payload = connection.recv()
        if command == 'close':
            return
        if command != 'initialize':
            raise ValueError('A cascade worker must be initialized before reset')
        data, physical, bins, days, gamma, trace, seed = payload
        connection.send((True, dict(initialized=True)))
        while True:
            command, value = connection.recv()
            if command == 'close':
                return
            try:
                if command == 'reset':
                    env = metrics = energy = None
                    years, episode = value
                    env = episode_environment(data, physical, years, days, bins)
                    metrics, energy = EpisodeRewardMetrics(gamma, action_correction_config=physical), np.zeros(5)
                    result = dict(obs=env.actor_observations(), state=env.state(), pid=os.getpid())
                elif command == 'step':
                    if env is None or env.index >= env.stop:
                        raise RuntimeError('Explicit reset is required before stepping an episode')
                    env.step(value)
                    transition = env.last_transition
                    metrics.update(transition)
                    energy += transition['energy']
                    done = env.index == env.stop
                    result = dict(obs=env.actor_observations(), state=env.state(),
                                  transition=transition, done=done)
                    if done:
                        record = episode_record(env, metrics, energy, seed, episode,
                                                gamma, trace, 0)
                        record['worker_pid'] = os.getpid()
                        result.update(episode_record=record, reward_metrics=metrics.summary())
                else:
                    raise ValueError('Unknown cascade worker command')
                connection.send((True, result))
            except Exception as exc:
                failure = None
                if env is not None and metrics is not None:
                    try:
                        if isinstance(exc, InfeasibleCascade):
                            metrics.record_execution_infeasibility(getattr(exc, 'reservoir_index', None))
                        failure = episode_record(env, metrics, energy, seed, episode, gamma, trace, 1,
                            dict(error_type=type(exc).__name__, reason=str(exc),
                                 diagnostics=getattr(exc, 'diagnostics', None)))
                    except Exception:
                        pass  # Preserve the original worker error if accounting also fails.
                connection.send((False, dict(error=repr(exc), traceback=traceback.format_exc(),
                                             episode_record=failure)))
    except (EOFError, BrokenPipeError):
        pass
    finally:
        connection.close()


class ParallelCascadeEnvs:
    """Send all lane commands before waiting; each lane owns a separate process."""

    def __init__(self, data, physical, bins, days, count, gamma, trace, seed):
        self.connections, self.processes = [], []
        self.closed = False
        context = mp.get_context('spawn')
        try:
            for _ in range(positive_integer(count, 'parallel_envs')):
                parent, child = context.Pipe()
                self.connections.append(parent)
                process = context.Process(target=_worker, args=(child,), daemon=True)
                try:
                    process.start()
                finally:
                    child.close()
                self.processes.append(process)
            for connection in self.connections:
                if not connection.poll(120):
                    raise RuntimeError('Cascade worker did not complete startup within 120 seconds')
                ok, result = connection.recv()
                if not ok or not result.get('ready'):
                    raise RuntimeError('Cascade worker did not report a valid startup message')
            self._exchange('initialize', [(data, physical, bins, days, gamma, trace, seed)]*len(self.connections))
        except (EOFError, BrokenPipeError, OSError) as exc:
            self.close()
            raise RuntimeError('Cascade worker failed during startup') from exc
        except BaseException:
            self.close()
            raise

    def _exchange(self, command, values):
        if self.closed or len(values) != len(self.connections):
            raise ValueError('Commands must match the live parallel environment count')
        replies = []
        try:
            for connection, value in zip(self.connections, values):
                connection.send((command, value))
            for connection in self.connections:
                if not connection.poll(120):
                    raise RuntimeError('Cascade worker did not respond within 120 seconds')
                replies.append(connection.recv())
        except (EOFError, BrokenPipeError, OSError, RuntimeError) as exc:
            self.close()
            error = RuntimeError('Cascade worker disconnected or timed out')
            error.successful_step_results = [(lane, value) for lane, (ok, value) in enumerate(replies) if ok]
            error.episode_records = [value['episode_record'] for ok, value in replies
                                    if not ok and value.get('episode_record') is not None]
            raise error from exc
        failures = [value for ok, value in replies if not ok]
        if failures:
            error = RuntimeError('Parallel cascade environment failed: '+failures[0]['error']+
                                 '\n'+failures[0]['traceback'])
            error.episode_records = [r['episode_record'] for r in failures if r['episode_record'] is not None]
            # Other lanes can finish this day's transition before a peer fails.
            # Let the parent persist those completed steps before stopping.
            error.successful_step_results = [(lane, value) for lane, (ok, value) in enumerate(replies) if ok]
            self.close()
            raise error
        return [value for _, value in replies]

    def reset(self, windows, episode_ids):
        return self._exchange('reset', list(zip(windows, episode_ids)))

    def step(self, actions):
        return self._exchange('step', list(actions))

    def close(self):
        if self.closed:
            return
        self.closed = True
        for connection in self.connections:
            try:
                connection.send(('close', None))
            except (BrokenPipeError, EOFError, OSError):
                pass
        for process in self.processes:
            process.join(timeout=2)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2)
        for connection in self.connections:
            connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *unused):
        self.close()
