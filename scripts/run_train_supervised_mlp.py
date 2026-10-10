"""Train the CWoLa MLP with true binary background/signal labels.

All data, fraction, backbone, optimizer, validation and checkpoint conventions
are shared with run_train_cwola. The signal fraction remains relative to the
mixture half, so the whole-batch signal fraction is approximately alpha / 2.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_train_cwola import main


if __name__ == "__main__":
    main(supervised=True)
