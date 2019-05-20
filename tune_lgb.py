from src.train.train_LGB import trainLGB

import sys

param = sys.argv[1]
trainLGB('cv', n_estimators=param)