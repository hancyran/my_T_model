setsid python train_LGB_bg.py cv 0.2000> /jet/workspace/2019Tencent/log/cv-0.txt 2>&1 &
setsid python train_LGB_bg.py val 0.2000> /jet/workspace/2019Tencent/log/val-0.txt 2>&1 &
setsid python train_LGB_bg.py cv 0.0500> /jet/workspace/2019Tencent/log/cv-1.txt 2>&1 &
setsid python train_LGB_bg.py val 0.0500> /jet/workspace/2019Tencent/log/val-1.txt 2>&1 &
