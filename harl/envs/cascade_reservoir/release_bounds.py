"""Validated dated proposal intervals; these are not additional physical limits."""
import copy
import csv
import hashlib
import io
from pathlib import Path

import numpy as np

from harl.envs.cascade_reservoir.data_loader import IDS

RELEASE_BOUNDS_FILE = 'five_reservoir_daily_release_bounds_1965_2024.csv'
RELEASE_MAPPING_MODES = ('joint_feasible', 'csv_proposal')
RELEASE_BOUNDS_COLUMNS = ('action_date', 'reservoir_id', 'Q_min_m3s', 'Q_max_m3s')


def validate_release_mapping_mode(value):
    if not isinstance(value, str) or value not in RELEASE_MAPPING_MODES:
        raise ValueError('release_mapping_mode must be joint_feasible or csv_proposal')
    return value


def validate_release_bounds_file(value):
    if (not isinstance(value, (str, Path)) or not str(value).strip() or
            Path(value).name != str(value) or str(value) in ('.','..')):
        raise ValueError('release_bounds_file must be a filename within the reservoir data directory')
    return str(value)


def _distribution(values):
    values = np.asarray(values, dtype=np.float64)
    scale = float(values.max())
    mean = 0. if scale == 0. else float((values/scale).mean()*scale)
    return dict(min=float(values.min()), max=scale, mean=mean,
        quantiles=dict(zip(('p00', 'p25', 'p50', 'p75', 'p90', 'p95', 'p99', 'p100'),
            np.quantile(values, (0., .25, .5, .75, .9, .95, .99, 1.)).tolist())))


class ReleaseBoundsData:
    """Read each source (action date, reservoir) exactly once and align to dates.

    All expected dates, including leap days, must have all five reservoir rows.
    Row order is immaterial. Zero lower bounds and degenerate intervals are
    accepted; negative, nonfinite or reversed bounds are rejected. Bounds above
    outlet capacity are diagnosed, never clipped or installed as hard limits.

    Column names declare m3/s. No release-bound-specific derivation or provenance
    sidecar is supplied, so loading cannot independently authenticate their
    operational meaning, derivation, or physical attainability.
    """
    def __init__(self, data, filename=RELEASE_BOUNDS_FILE):
        self.file = validate_release_bounds_file(filename)
        self.path = (Path(data.root)/self.file).resolve()
        if not self.path.is_file():
            raise ValueError('CSV proposal mapping requires '+str(self.path))
        self.data = data
        self.dates = tuple(data.dates)
        if not self.dates or len(set(self.dates)) != len(self.dates):
            raise ValueError('Release-bound alignment requires nonempty unique action dates')
        self.date_index = {day:index for index,day in enumerate(self.dates)}
        by_stamp = {day.isoformat():index for index,day in enumerate(self.dates)}
        by_reservoir = {name:index for index,name in enumerate(IDS)}
        shape = (len(self.dates), len(IDS))
        self.low, self.high = np.empty(shape), np.empty(shape)
        seen = np.zeros(shape, dtype=bool)
        payload = self.path.read_bytes()
        self.sha256 = hashlib.sha256(payload).hexdigest()
        try:
            source = payload.decode('utf-8-sig')
        except UnicodeDecodeError as exc:
            raise ValueError('Release-bound CSV must be UTF-8') from exc
        reader = csv.DictReader(io.StringIO(source, newline=''))
        headers = reader.fieldnames or []
        if len(headers) != len(set(headers)) or set(headers) != set(RELEASE_BOUNDS_COLUMNS):
            raise ValueError('Release-bound CSV requires exactly '+','.join(RELEASE_BOUNDS_COLUMNS))
        for row in reader:
            if None in row or any(row.get(k) is None or not row[k].strip() for k in RELEASE_BOUNDS_COLUMNS):
                raise ValueError('Missing release-bound value at row '+str(reader.line_num))
            stamp, name = row['action_date'].strip(), row['reservoir_id'].strip()
            if stamp not in by_stamp or name not in by_reservoir:
                raise ValueError('Unknown release-bound action date/reservoir: '+stamp+' '+name)
            t, i = by_stamp[stamp], by_reservoir[name]
            if seen[t, i]:
                raise ValueError('Duplicate release-bound action date/reservoir: '+stamp+' '+name)
            try:
                low, high = float(row['Q_min_m3s']), float(row['Q_max_m3s'])
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError('Invalid release-bound number at row '+str(reader.line_num)) from exc
            if not np.isfinite((low, high)).all() or low < 0. or high < 0. or low > high:
                raise ValueError('Release bounds must be finite, nonnegative and ordered at row '+str(reader.line_num))
            self.low[t, i], self.high[t, i] = low, high
            seen[t, i] = True
        if not seen.all():
            t, i = np.argwhere(~seen)[0]
            raise ValueError('Release bounds must cover every actual action date and reservoir; missing '+
                             self.dates[t].isoformat()+' '+IDS[i])
        capacity = np.asarray(data.release_cap, dtype=np.float64)
        if capacity.shape != (5,) or not np.isfinite(capacity).all() or (capacity <= 0.).any():
            raise ValueError('Release-bound diagnostics require positive finite outlet capacities')
        self.release_cap = capacity.copy()
        self.release_cap.setflags(write=False)
        self.low.setflags(write=False)
        self.high.setflags(write=False)
        self.source_date_min = min(self.dates).isoformat()
        self.source_date_max = max(self.dates).isoformat()

    def take(self, indices, data=None):
        """Build an aligned read-only view without reading or reindexing the CSV."""
        indices = np.asarray(indices)
        if (indices.ndim != 1 or not len(indices) or indices.dtype.kind not in 'iu' or
                (indices < 0).any() or (indices >= len(self.dates)).any() or
                len(np.unique(indices)) != len(indices)):
            raise ValueError('Release-bound view requires unique valid integer source indices')
        result = copy.copy(self)
        result.data = self.data if data is None else data
        result.dates = tuple(self.dates[i] for i in indices)
        result.date_index = {day:i for i,day in enumerate(result.dates)}
        result.low, result.high = self.low[indices].copy(), self.high[indices].copy()
        result.low.setflags(write=False)
        result.high.setflags(write=False)
        return result

    def bounds_for_date(self, day):
        """Return copies by an actual action date, independent of episode offsets."""
        if isinstance(day, str):
            from datetime import date
            try:
                day = date.fromisoformat(day)
            except ValueError as exc:
                raise ValueError('Invalid release-bound action date') from exc
        if day not in self.date_index:
            raise ValueError('Action date outside this release-bound view: '+str(day))
        index = self.date_index[day]
        return self.low[index].copy(), self.high[index].copy()

    def metadata(self):
        width = self.high-self.low
        by = {}
        for i,name in enumerate(IDS):
            by[name] = dict(proposal_low_m3s=_distribution(self.low[:, i]),
                proposal_high_m3s=_distribution(self.high[:, i]),
                proposal_width_m3s=_distribution(width[:, i]),
                degenerate_interval_count=int((width[:, i] == 0.).sum()),
                low_exceeds_outlet_capacity_count=int((self.low[:, i] > self.release_cap[i]).sum()),
                high_exceeds_outlet_capacity_count=int((self.high[:, i] > self.release_cap[i]).sum()),
                max_upper_capacity_excess_m3s=float(np.maximum(0.,self.high[:, i]-self.release_cap[i]).max()))
        return dict(schema='cascade_release_proposals_v1', file=self.file, source_path=str(self.path),
            sha256=self.sha256, reservoir_ids=list(IDS), shape=list(self.low.shape),
            action_days=len(self.dates), reservoir_days=int(self.low.size),
            first_action_date=self.dates[0].isoformat(), last_action_date=self.dates[-1].isoformat(),
            source_date_min=self.source_date_min, source_date_max=self.source_date_max,
            units='m3/s', units_basis='CSV column names Q_min_m3s and Q_max_m3s',
            time_basis='action_date; no next-day shift', role='proposal_mapping_interval_only',
            physical_hard_constraint=False,
            release_semantics='total reservoir release (user-declared), not turbine-only flow',
            derivation='user reports historical inflows and admissible storage envelopes; generator not supplied',
            information_limitation='historical realized daily inflows may not be known before dispatch; causal/forecast availability and use of future data cannot be verified without the construction algorithm',
            provenance_status='supplied CSV; derivation and operational semantics not independently verified',
            diagnostics=dict(proposal_width_m3s=_distribution(width),
                degenerate_interval_count=int((width == 0.).sum()),
                low_exceeds_outlet_capacity_count=int((self.low > self.release_cap).sum()),
                high_exceeds_outlet_capacity_count=int((self.high > self.release_cap).sum()),
                by_reservoir=by))
