import os

from src.train.train_LGB import trainLGB
from src.utils.LGB_args import args

n_estimator_list = args.n_estimators

for n in n_estimator_list:
    trainLGB('val', n_estimators=n)

print('All Finished')