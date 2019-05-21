import sys

from src.train.train_LGB import trainLGB
from src.utils.LGB_args import args

# train_type = sys.argv[1]
# n = float(sys.argv[2])

learning_rate_list = args.learning_rate

for i, n in enumerate(learning_rate_list):
    # f.write('setsid python train_LGB_bg.py cv %.4f> %s 2>&1 &\n' % (n, getLogPath('cv', i)))
    # f.write('setsid python train_LGB_bg.py val %.4f> %s 2>&1 &\n' % (n, getLogPath('val', i)))
    print('Time %d' %i)
    trainLGB('cv', learning_rate=n)
    trainLGB('val', learning_rate=n)

