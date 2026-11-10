"""Dated CSV proposal bounds: strict coverage, diagnostics and episode alignment."""
import copy
import csv
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harl.envs.cascade_reservoir.data_loader import CascadeData, IDS
from harl.envs.cascade_reservoir.episodes import EpisodeData
from harl.envs.cascade_reservoir.release_bounds import (
    RELEASE_BOUNDS_COLUMNS, RELEASE_BOUNDS_FILE, ReleaseBoundsData,
    validate_release_mapping_mode)


def rows_for(dates):
    return [dict(action_date=day.isoformat(), reservoir_id=name,
                 Q_min_m3s=t*5+i, Q_max_m3s=t*5+i+2)
            for t,day in enumerate(dates) for i,name in enumerate(IDS)]


def write_rows(root, rows, columns=RELEASE_BOUNDS_COLUMNS):
    path = Path(root)/RELEASE_BOUNDS_FILE
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows([[row.get(key, '') for key in columns] for row in rows])
    return path


def small_data(root, dates):
    return SimpleNamespace(root=Path(root), dates=tuple(dates), release_cap=np.full(5,10.))


class ReleaseBoundsValidationTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(dir=str(ROOT/'tests'))
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.dates = (date(2000,2,28), date(2000,2,29), date(2000,3,1))
        self.data = small_data(self.root,self.dates)

    def test_leap_date_and_arbitrary_csv_order_align_by_actual_date_and_id(self):
        rows = rows_for(self.dates)
        path = write_rows(self.root,list(reversed(rows)))
        bounds = ReleaseBoundsData(self.data)
        expected = np.arange(15,dtype=float).reshape(3,5)
        np.testing.assert_array_equal(bounds.low,expected)
        np.testing.assert_array_equal(bounds.high,expected+2.)
        self.assertEqual(bounds.dates,self.dates)
        self.assertEqual(bounds.date_index[date(2000,2,29)],1)
        self.assertEqual(bounds.sha256,hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertFalse(bounds.low.flags.writeable)
        self.assertFalse(bounds.high.flags.writeable)
        low, high = bounds.bounds_for_date('2000-02-29')
        low[:] = -123.
        np.testing.assert_array_equal(bounds.low[1],expected[1])
        np.testing.assert_array_equal(high,expected[1]+2.)
        with self.assertRaises(ValueError):
            bounds.bounds_for_date('2001-02-29')
        with self.assertRaises(ValueError):
            bounds.bounds_for_date(date(2000,3,2))

    def test_missing_duplicate_unknown_date_and_unknown_reservoir_are_rejected(self):
        cases = []
        base = rows_for(self.dates)
        cases.append(base[:-1])
        cases.append(base+[dict(base[0])])
        bad_date = copy.deepcopy(base)
        bad_date[0]['action_date'] = '2000-03-02'
        cases.append(bad_date)
        bad_name = copy.deepcopy(base)
        bad_name[0]['reservoir_id'] = 'TGR'
        cases.append(bad_name)
        for rows in cases:
            with self.subTest(rows=rows[:1]):
                write_rows(self.root,rows)
                with self.assertRaises(ValueError):
                    ReleaseBoundsData(self.data)

    def test_missing_extra_duplicate_schema_and_blank_values_are_rejected(self):
        for columns in (RELEASE_BOUNDS_COLUMNS[:-1], RELEASE_BOUNDS_COLUMNS+('unknown',),
                        RELEASE_BOUNDS_COLUMNS+('Q_min_m3s',)):
            with self.subTest(columns=columns):
                write_rows(self.root,rows_for(self.dates),columns)
                with self.assertRaises(ValueError):
                    ReleaseBoundsData(self.data)
        rows = rows_for(self.dates)
        rows[3]['Q_max_m3s'] = ''
        write_rows(self.root,rows)
        with self.assertRaises(ValueError):
            ReleaseBoundsData(self.data)

    def test_negative_nonfinite_nonnumeric_and_reversed_bounds_are_rejected(self):
        for low, high in ((-1.,2.),(1.,-2.),('nan',2.),(1.,'inf'),('bad',2.),(3.,2.)):
            with self.subTest(low=low,high=high):
                rows = rows_for(self.dates)
                rows[0].update(Q_min_m3s=low,Q_max_m3s=high)
                write_rows(self.root,rows)
                with self.assertRaises(ValueError):
                    ReleaseBoundsData(self.data)

    def test_capacity_exceedance_and_degeneracy_are_diagnostics_without_clipping(self):
        rows = rows_for(self.dates)
        rows[0].update(Q_min_m3s=0.,Q_max_m3s=0.)
        rows[1].update(Q_min_m3s=25.,Q_max_m3s=30.)
        write_rows(self.root,rows)
        bounds = ReleaseBoundsData(self.data)
        self.assertEqual(bounds.low[0,1],25.)
        self.assertEqual(bounds.high[0,1],30.)
        metadata = bounds.metadata()
        self.assertFalse(metadata['physical_hard_constraint'])
        self.assertEqual(metadata['units'],'m3/s')
        self.assertIn('not independently verified',metadata['provenance_status'])
        by = metadata['diagnostics']['by_reservoir']
        self.assertEqual(by['WDD']['degenerate_interval_count'],1)
        self.assertEqual(by['BHT']['max_upper_capacity_excess_m3s'],20.)
        self.assertGreaterEqual(by['BHT']['low_exceeds_outlet_capacity_count'],1)
        self.assertEqual(by['WDD']['proposal_width_m3s']['min'],0.)
        json.dumps(metadata,allow_nan=False)

    def test_episode_view_preserves_nonconsecutive_year_order_leap_day_and_offsets(self):
        begin = date(1999,1,1)
        days = tuple(begin+timedelta(days=i) for i in range(365+366+365))
        source = small_data(self.root,days)
        source.external = np.zeros((len(days),5))
        source.indices = lambda first,last: (
            next(i for i,day in enumerate(days) if day.year == first),
            max(i for i,day in enumerate(days) if day.year == last)+1)
        write_rows(self.root,rows_for(days))
        source.release_bounds = ReleaseBoundsData(source)
        source.load_release_bounds = mock.Mock(return_value=source.release_bounds)
        view = EpisodeData(source,[2000,1999],370)
        self.assertEqual(view.dates[59],date(2000,2,29))
        self.assertEqual(view.dates[366],date(1999,1,1))
        np.testing.assert_array_equal(view.release_bounds.low,
            source.release_bounds.low[view.source_indices])
        self.assertEqual(view.release_bounds.dates,view.dates)
        self.assertEqual(view.release_bounds.date_index[date(1999,1,1)],366)
        self.assertIs(view.load_release_bounds(),view.release_bounds)
        source.load_release_bounds.assert_not_called()
        offset = view.release_bounds.take(np.array([58,59,60]))
        self.assertEqual(offset.dates,(date(2000,2,28),date(2000,2,29),date(2000,3,1)))
        np.testing.assert_array_equal(offset.bounds_for_date('2000-02-29')[0],
                                      source.release_bounds.low[424])
        for invalid in ([], [1,1], [-1], [370], [1.5]):
            with self.subTest(indices=invalid), self.assertRaises(ValueError):
                view.release_bounds.take(invalid)


class ProductionReleaseBoundsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = yaml.safe_load((ROOT/'harl/configs/envs_cfgs/cascade_reservoir.yaml').read_text())
        cls.config.update(head_model='fixed',use_daily_level_limits=False,
                          load_service_requirements=False,release_mapping_mode='csv_proposal')
        cls.data = CascadeData(ROOT/'data/cascade_reservoir_new',cls.config)

    def test_complete_1965_2024_coverage_and_cached_loading(self):
        bounds = self.data.release_bounds
        self.assertEqual(bounds.low.shape,(21915,5))
        self.assertEqual(bounds.high.shape,(21915,5))
        self.assertEqual(len(bounds.date_index),21915)
        self.assertIn(date(1968,2,29),bounds.date_index)
        self.assertIn(date(2024,2,29),bounds.date_index)
        leap_index = bounds.date_index[date(2024,2,29)]
        self.assertEqual(bounds.dates[leap_index-1],date(2024,2,28))
        self.assertEqual(bounds.dates[leap_index+1],date(2024,3,1))
        self.assertEqual(bounds.dates[0],date(1965,1,1))
        self.assertEqual(bounds.dates[-1],date(2024,12,31))
        self.assertEqual(self.data.hashes[RELEASE_BOUNDS_FILE],bounds.sha256)
        with mock.patch.object(Path,'read_bytes',side_effect=AssertionError('Must not reload')):
            self.assertIs(self.data.load_release_bounds(),bounds)
        self.assertEqual(bounds.metadata()['reservoir_days'],109575)

    def test_joint_feasible_default_does_not_read_or_require_release_csv(self):
        config = copy.deepcopy(self.config)
        config.pop('release_mapping_mode')
        config['release_bounds_file'] = 'deliberately_missing_proposal_file.csv'
        with mock.patch('harl.envs.cascade_reservoir.release_bounds.ReleaseBoundsData',
                        side_effect=AssertionError('Baseline must not load proposal bounds')):
            data = CascadeData(ROOT/'data/cascade_reservoir_new',config)
        self.assertEqual(data.release_mapping_mode,'joint_feasible')
        self.assertIsNone(data.release_bounds)
        self.assertNotIn(config['release_bounds_file'],data.hashes)
        with self.assertRaises(ValueError):
            data.load_release_bounds()

    def test_missing_csv_and_invalid_mapping_mode_fail_explicitly(self):
        config = copy.deepcopy(self.config)
        config['release_bounds_file'] = 'deliberately_missing_proposal_file.csv'
        with self.assertRaises(ValueError):
            CascadeData(ROOT/'data/cascade_reservoir_new',config)
        for value in ('quadratic','csv',False,None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_release_mapping_mode(value)


if __name__ == '__main__':
    unittest.main()
