import sys

from src.train.train_LGB import trainLGB

train_type = sys.argv[1]

try:
    learning_rate = int(sys.argv[2])
    if train_type == 'cv':
        trainLGB('cv', learning_rate=learning_rate)
    elif train_type == 'val':
        trainLGB('val', learning_rate=learning_rate)
except:
    if train_type == 'test':
        trainLGB('test')
