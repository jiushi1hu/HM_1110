"""Local observations and current-transition physics for the reservoir cascade."""
import calendar
import numpy as np
from gym.spaces import Box, Discrete
from harl.envs.cascade_reservoir.reservoir_physics import ReservoirPhysics, InfeasibleCascade
from harl.envs.cascade_reservoir.reward import OperationReward, reward_settings
from harl.envs.cascade_reservoir.episodes import (
    DEFAULT_EPISODE_DAYS, DEFAULT_ACTION_BINS, positive_integer)

ACTOR_OBSERVATION_LAYOUT = 'local_neighbors_next_day_limits_v2'
LOCAL_NEIGHBOR_FEATURES = (
    'storage_fraction', 'external_inflow_standardized', 'previous_release_fraction',
    'upstream_storage_fraction', 'upstream_external_inflow_standardized',
    'upstream_previous_release_fraction', 'downstream_storage_fraction',
    'downstream_headroom_fraction',
)
LOCAL_RULE_FEATURES = (
    'current_state_storage_lower_fraction', 'current_state_storage_upper_fraction',
    'next_day_level_lower_fraction', 'next_day_level_upper_fraction',
)
TIME_FEATURES = ('season_sin', 'season_cos', 'episode_time_remaining_fraction')


def actor_observation_features(use_daily_level_limits):
    return LOCAL_NEIGHBOR_FEATURES + (LOCAL_RULE_FEATURES if use_daily_level_limits else ()) + TIME_FEATURES


class CascadeReservoirEnv:
    n_agents = 5
    observation_layout = ACTOR_OBSERVATION_LAYOUT

    def observation_metadata(self):
        return dict(actor_observation_layout=self.observation_layout,
                    actor_observation_size=self.observation_size, critic_state_size=self.state_size)

    def __init__(self, data, config, start_year=None, action_bins=DEFAULT_ACTION_BINS,
                 episode_days=DEFAULT_EPISODE_DAYS):
        self.data = data
        self.physics = ReservoirPhysics(data, config)
        self.release_bounds = getattr(data, 'release_bounds', None)
        if self.physics.release_mapping_mode == 'csv_proposal' and self.release_bounds is None:
            raise ValueError('csv_proposal requires release bounds loaded by CascadeData')
        self.use_daily_level_limits = config.get('use_daily_level_limits', True)
        if not isinstance(self.use_daily_level_limits, bool):
            raise ValueError('use_daily_level_limits must be a boolean')
        self.level_limits = data.load_daily_level_limits() if self.use_daily_level_limits else None
        self.reward_settings = reward_settings(config)
        self.operation_reward = OperationReward(data, self.reward_settings,
                                               timestep_seconds=self.physics.dt)
        start_year = data.dates[0].year if start_year is None else int(start_year)
        self.episode_days = positive_integer(episode_days, 'episode_days')
        self.start, _ = data.indices(start_year, start_year)
        self.stop = self.start+self.episode_days
        if self.stop > len(data.dates):
            raise ValueError('Episode extends beyond the available daily data')
        self.time_scale_days = self.episode_days
        self._storage_span = data.vmax-data.vmin
        self._level_min, self._level_max = data.levels(data.vmin), data.levels(data.vmax)
        self._level_span = self._level_max-self._level_min
        if (self._storage_span <= 0.).any() or (self._level_span <= 0.).any():
            raise ValueError('Operating storage and water-level ranges must be positive')
        self.observation_features = actor_observation_features(self.use_daily_level_limits)
        self.observation_size = len(self.observation_features)
        self.state_size = 38 if self.use_daily_level_limits else 18
        self.observation_space = [Box(-np.inf, np.inf, (self.observation_size,), dtype=np.float32) for _ in range(5)]
        self.share_observation_space = [Box(-np.inf, np.inf, (self.state_size,), dtype=np.float32) for _ in range(5)]
        if (isinstance(action_bins, bool) or
                not isinstance(action_bins, (int, np.integer)) or action_bins < 2):
            raise ValueError('action_bins must be an integer >= 2')
        self.action_bins = action_bins
        self.action_space = [Discrete(action_bins) for _ in range(5)]
        self.reset()

    def _current_state_storage_bounds(self, index):
        """The previous executed row constrains the current state; reset uses global limits."""
        if self.level_limits is None or index == self.start:
            return self.data.vmin, self.data.vmax
        return self.level_limits.storage_lower[index-1], self.level_limits.storage_upper[index-1]

    def _check_current_storage(self, storage, index):
        if storage.shape != (5,) or not np.isfinite(storage).all():
            raise InfeasibleCascade('Starting storage must be a finite five-vector')
        lower, upper = self._current_state_storage_bounds(index)
        tolerance = np.maximum(1., 1e-10*np.abs(storage))
        if ((storage < lower-tolerance) | (storage > upper+tolerance)).any():
            raise InfeasibleCascade('Starting storage violates the preceding action\'s bounds')

    def reset(self):
        storage = self.data.initial_storage(self.data.dates[self.start].year)
        self._check_current_storage(storage, self.start)
        self.index = self.start
        self.storage = storage
        self.previous_release = np.zeros(5)
        self.last_transition = None
        return self._observations()

    def state(self):
        """Centralized state with known next-state limits, without future forcing."""
        if self.index == self.stop:
            return np.zeros(self.state_size, dtype=np.float32)
        day = self.data.dates[self.index]
        phase = 2.*np.pi*(day.timetuple().tm_yday-1)/(366 if calendar.isleap(day.year) else 365)
        parts = [
            (self.storage-self.data.vmin)/self._storage_span,
            (self.data.external[self.index]-self.data.flow_mean)/self.data.flow_std,
            self.previous_release/self.data.release_cap,
            [np.sin(phase),np.cos(phase)],
            [(self.stop-self.index)/self.time_scale_days],
        ]
        if self.level_limits is not None:
            lower, upper = self._current_state_storage_bounds(self.index)
            # The CSV row for action date t already describes H[t+1]. There
            # is no extra date shift, even at a source-year or episode boundary.
            parts.extend([(lower-self.data.vmin)/self._storage_span,
                          (upper-self.data.vmin)/self._storage_span,
                          (self.level_limits.level_lower[self.index]-self._level_min)/self._level_span,
                          (self.level_limits.level_upper[self.index]-self._level_min)/self._level_span])
        return np.concatenate(parts).astype(np.float32)

    def actor_observations(self):
        """Own state, immediate neighbors, current forcing and next-state limits.

        Reservoirs choose actions before upstream releases are resolved, so
        the upstream release feature is yesterday's release. Downstream
        headroom is signed room below the upper bound for the next state.
        """
        if self.index == self.stop:
            return np.zeros((5,self.observation_size),dtype=np.float32)
        state = self.state()
        storage, external, release = state[:5], state[5:10], state[10:15]
        upper = (np.ones(5) if self.level_limits is None else
                 (self.level_limits.storage_upper[self.index]-self.data.vmin)/self._storage_span)
        observations = []
        for i in range(5):
            upstream = (storage[i-1],external[i-1],release[i-1]) if i else (0.,0.,0.)
            downstream = (storage[i+1],upper[i+1]-storage[i+1]) if i < 4 else (0.,1.)
            local = [storage[i],external[i],release[i],*upstream,*downstream]
            if self.level_limits is not None:
                local.extend(state[start+i] for start in (18,23,28,33))
            local.extend(state[15:18])
            observations.append(local)
        return np.asarray(observations,dtype=np.float32)

    def _observations(self):
        state = np.repeat(self.state()[None,:],5,axis=0)
        return self.actor_observations(), state, None

    def normalize_actions(self, actions):
        actions = np.asarray(actions,dtype=np.float64)
        if actions.shape not in ((5,),(5,1)) or not np.isfinite(actions).all():
            raise ValueError('Five finite scalar actions required')
        actions = actions.reshape(5)
        if ((actions != np.floor(actions)) | (actions < 0) | (actions >= self.action_bins)).any():
            raise ValueError('Discrete actions must be integer indices in [0,action_bins-1]')
        return actions/(self.action_bins-1)

    def step(self, actions):
        if self.index >= self.stop:
            raise RuntimeError('Episode finished; reset explicitly')
        actions = np.asarray(actions)
        normalized = self.normalize_actions(actions)
        try:
            self._check_current_storage(self.storage,self.index)
            bounds = {} if self.level_limits is None else dict(
                storage_lower=self.level_limits.storage_lower[self.index],
                storage_upper=self.level_limits.storage_upper[self.index])
            if self.physics.release_mapping_mode == 'csv_proposal':
                # Resolve by the actual action date, including sliced/reordered episodes.
                row = self.release_bounds.date_index[self.data.dates[self.index]]
                bounds.update(proposal_low=self.release_bounds.low[row],
                              proposal_high=self.release_bounds.high[row])
            result = self.physics.transition(self.storage,self.data.external[self.index],normalized,**bounds)
        except RuntimeError as exc:
            exc.args = ('{}; date={} storage_m3={} actions={}'.format(
                exc,self.data.dates[self.index],self.storage.tolist(),actions.reshape(5).tolist()),)
            if isinstance(getattr(exc, 'diagnostics', None), dict):
                exc.diagnostics.update(date=self.data.dates[self.index].isoformat(),
                                       sampled_actions=actions.reshape(5).tolist())
            raise
        result['date'] = self.data.dates[self.index].isoformat()
        result['sampled_actions'] = actions.reshape(5).astype(np.int64).copy()
        result['action_bins'] = self.action_bins
        result['timestep_seconds'] = self.physics.dt
        result['daily_level_limits'] = self.use_daily_level_limits
        # The policy proposal remains distinct from realized/forced dispatch.
        result['normalized_actions'] = normalized.copy()
        result['actions'] = np.where(result['controls'] > 0,actions.reshape(5),0).astype(np.int64)
        result['levels_start'] = self.physics._forebay_levels(result['storage_start'])
        result['levels_end'] = self.data.levels(result['storage_end'])
        if self.level_limits is None:
            result.update(storage_lower_m3=self.data.vmin.copy(), storage_upper_m3=self.data.vmax.copy(),
                          next_state_level_lower_m=self._level_min.copy(),
                          next_state_level_upper_m=self._level_max.copy())
        self.operation_reward.apply(result)
        infos = [dict(bad_transition=False,use_daily_level_limits=self.use_daily_level_limits) for _ in range(5)]
        if self.level_limits is not None:
            result.update(next_state_level_lower_m=self.level_limits.level_lower[self.index].copy(),
                          next_state_level_upper_m=self.level_limits.level_upper[self.index].copy())
            for j,info in enumerate(infos):
                info.update(storage_lower_m3=float(self.level_limits.storage_lower[self.index,j]),
                            storage_upper_m3=float(self.level_limits.storage_upper[self.index,j]),
                            next_state_level_lower_m=float(self.level_limits.level_lower[self.index,j]),
                            next_state_level_upper_m=float(self.level_limits.level_upper[self.index,j]))
        self.last_transition = result
        self.storage = result['storage_end'].copy()
        self.previous_release = result['release'].copy()
        self.index += 1
        done = self.index == self.stop
        obs,state,available = self._observations()
        return obs,state,result['mixed_rewards'].astype(np.float32).reshape(5,1), \
            np.full(5,done,dtype=bool),infos,available
