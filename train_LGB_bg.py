import sys

from src.train.train_LGB import trainLGB
from src.utils.LGB_args import args

for i, n in enumerate(args.learning_rate):
    print('CV Time: %d' % i)
    trainLGB('cv', learning_rate=n)
    print('Val Time: %d' % i)
    trainLGB('val', learning_rate=n)
