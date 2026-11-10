"""Check the interactive Matplotlib backend used by the training entry point."""
import argparse
import importlib
import sys

import matplotlib


def select_interactive_backend():
    backend = 'MacOSX' if sys.platform == 'darwin' else 'TkAgg'
    matplotlib.use(backend, force=True)
    importlib.import_module('matplotlib.backends.backend_'+backend.lower())
    return backend


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-backend', action='store_true', required=True,
                        help='Check the interactive plotting backend without opening a window')
    parser.parse_args()
    backend = select_interactive_backend()
    print('Interactive plotting backend available: '+backend, flush=True)


if __name__ == '__main__':
    main()
