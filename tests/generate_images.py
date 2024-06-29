import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from model.generate import sample

sample()