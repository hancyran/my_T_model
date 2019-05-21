import sys

from src.train.train_LGB import trainLGB
from src.utils.LGB_args import args

train_type = sys.argv[1]
n = float(sys.argv[2])

if train_type == 'val':
    trainLGB(train_type, learning_rate=n)

