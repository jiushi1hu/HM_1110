"""Replay a saved cascade policy without training or changing source datasets."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harl.envs.cascade_reservoir.checkpoint import load_checkpoint, ReplayTraceWriter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True,
                        help='Trusted local evaluation checkpoint produced by train_cascade.py')
    parser.add_argument('--source-years', nargs='+', type=int, required=True,
                        help='Ordered source years containing the requested replay horizon')
    parser.add_argument('--episode-days', type=int,
                        help='Must equal the saved training horizon; omit to use the saved value')
    parser.add_argument('--data-dir', type=Path,
                        help='Relocated source directory; data and implementation hashes must match')
    parser.add_argument('--override-action', nargs=3, action='append', default=[],
                        metavar=('STEP', 'RESERVOIR', 'INDEX'),
                        help='Replace one discrete command at a one-based step, e.g. 1 BHT 16; repeat as needed')
    parser.add_argument('--output', type=Path, required=True,
                        help='New CSV: five reservoir rows per day, including physical dispatch and saved values')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output must name a new file')
    overrides = {}
    try:
        for step, name, value in args.override_action:
            key = (int(step), name)
            if key in overrides:
                parser.error('Duplicate action override: '+str(key))
            overrides[key] = int(value)
    except ValueError:
        parser.error('Override step and action index must be integers')
    policy = load_checkpoint(args.checkpoint, data_root=args.data_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8', newline='') as stream:
        result = policy.replay(args.source_years, args.episode_days, ReplayTraceWriter(stream),
                               action_overrides=overrides)
    result['checkpoint'] = str(args.checkpoint.resolve())
    result['trace_csv'] = str(args.output.resolve())
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
