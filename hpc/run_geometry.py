"""Run geometry CLI with Slurm warning/termination signals checkpointed."""
from pathlib import Path
import signal
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scgvb.cli.molecule import main


def interrupt(signum, frame):
    print(f'Received signal {signum}; retaining completed geometry evaluations.', flush=True)
    raise KeyboardInterrupt


if __name__ == '__main__':
    for signum in (signal.SIGUSR1, signal.SIGTERM):
        signal.signal(signum, interrupt)
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print('Geometry interrupted. Resume with --geometry-restart-json and the saved output.', flush=True)
        raise SystemExit(130)
