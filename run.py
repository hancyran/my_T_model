import os

from src.utils.LGB_args import args
from src.utils.file_path_util import getLogPath
#
# n_estimator_list = args.n_estimators
#
# for i, n in enumerate(n_estimator_list):
#     print('CV Tuning %d\n' % i)
#     os.system('setsid python train_LGB_bg.py cv %d> %s 2>&1 &' % (n, getLogPath('cv')))
#
# for i, n in enumerate(n_estimator_list):
#     print('CV Tuning %d\n' % i)
#     os.system('setsid python train_LGB_bg.py val %d> %s 2>&1 &' % (n, getLogPath('val')))


learning_rate_list = args.learning_rate

with open('run.sh', 'w') as f:
    for i, n in enumerate(learning_rate_list):
        f.write('setsid python train_LGB_bg.py cv %.4f> %s 2>&1 &\n' % (n, getLogPath('cv')))
        f.write('setsid python train_LGB_bg.py val %.4f> %s 2>&1 &\n' % (n, getLogPath('val')))

