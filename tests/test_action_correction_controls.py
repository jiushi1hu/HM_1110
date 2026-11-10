"""CLI A/B/C controls and backward-compatible correction figures."""
import copy
from contextlib import redirect_stdout, redirect_stderr
import importlib.util
import io
import unittest
from unittest import mock

try:
    import matplotlib
    matplotlib.use('Agg', force=True)
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


@unittest.skipUnless(importlib.util.find_spec('torch'), 'Training interpreter requires torch')
class CorrectionControlsTest(unittest.TestCase):
    def configure(self, arguments):
        from examples import train_cascade as cli
        def capture(data, physical, config, options):
            return physical, config
        with mock.patch.object(cli, 'CascadeData', return_value=mock.sentinel.data), redirect_stdout(io.StringIO()):
            return cli.main(['--no-plot']+arguments, experiment_runner=capture)

    def test_abc_switches_preserve_service_weights_and_ppo_settings(self):
        from examples import train_cascade as cli
        baseline, physical = cli.load_configuration()
        for flags, expected in [
            (['--no-action-correction', '--no-action-correction-penalty'], (False, False)),
            (['--action-correction', '--no-action-correction-penalty'], (True, False)),
            (['--action-correction', '--action-correction-penalty'], (True, True))]:
            with self.subTest(flags=flags):
                result, algorithm = self.configure(flags)
                self.assertEqual((result['action_correction']['enabled'],
                                  result['action_correction']['penalty_enabled']), expected)
                self.assertEqual(result['action_correction']['tau'], .03)
                self.assertEqual(result['action_correction']['correction_weight'], 1.)
                self.assertEqual(result['reward_penalties'], physical['reward_penalties'])
                self.assertEqual(result['release_mapping'], 'linear')
                self.assertEqual(algorithm['algo'], baseline['algo'])
                self.assertEqual(algorithm['model'], baseline['model'])

    def test_threshold_and_weight_overrides_and_invalid_values(self):
        result, _ = self.configure(['--action-correction', '--action-correction-penalty',
                                   '--correction-tau', '.1', '--correction-weight', '2.5'])
        self.assertEqual(result['action_correction']['tau'], .1)
        self.assertEqual(result['action_correction']['correction_weight'], 2.5)
        for option, value in [('--correction-tau', '-.1'), ('--correction-tau', 'nan'),
                              ('--correction-weight', '-1'), ('--correction-weight', 'inf')]:
            with self.subTest(option=option, value=value), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    self.configure([option, value])


@unittest.skipIf(plt is None, 'Plotting interpreter requires matplotlib')
class CorrectionFigureTest(unittest.TestCase):
    def tearDown(self):
        plt.close('all')

    def payload(self):
        from test_cascade_comparison_plot import payload, use_current_learning_objectives
        data = use_current_learning_objectives(payload())
        for entry in data['runs']:
            for history in entry['payload']['evaluation_histories']:
                for row in history:
                    for scope in row['windows']:
                        scope.update(action_correction_violation_count=0,
                                     action_correction_active_reservoir_days=9,
                                     action_correction_violation_rate=0.)
                    row.update(action_correction_violation_count=0,
                               action_correction_active_reservoir_days=27,
                               action_correction_violation_rate=0.)
        return data

    def test_new_comparison_adds_third_axis_and_preserves_legacy_axes(self):
        from examples import plot_cascade_compare as compare
        from examples import plot_cascade_parallel as single
        data = self.payload()
        figures = compare.make_figures(data)
        self.assertEqual(len(figures['violations'].axes), 3)
        self.assertEqual(figures['violations'].axes[2].get_title(), 'Action-correction rate')
        for line in figures['violations'].axes[2].lines:
            self.assertTrue(all(value == 0 for value in line.get_ydata()))
        figure = single.make_violation_figure(data['runs'][0]['payload'])
        self.assertEqual(len(figure.axes), 3)

    def test_incomplete_metrics_and_invalid_denominator_are_rejected(self):
        from examples import plot_cascade_compare as compare
        for mutation in ('missing', 'denominator', 'rate'):
            data = self.payload()
            scope = data['runs'][0]['payload']['evaluation_histories'][0][0]['windows'][0]
            if mutation == 'missing':
                del scope['action_correction_violation_rate']
            elif mutation == 'denominator':
                scope['action_correction_active_reservoir_days'] = 11
            else:
                scope['action_correction_violation_rate'] = .2
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                compare.make_figures(data)

    def test_failed_evaluations_keep_correction_rates_null(self):
        from examples import plot_cascade_parallel as single
        from test_cascade_comparison_plot import evaluation
        row = evaluation(1, 1, 0., failed=True)
        for scope in [row]+row['windows']:
            scope.update(action_correction_violation_count=None,
                         action_correction_active_reservoir_days=None,
                         action_correction_violation_rate=None)
        self.assertTrue(single.validate_action_correction_metrics(row))


if __name__ == '__main__':
    unittest.main()
