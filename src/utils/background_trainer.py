import os


def bg_trainer():
    os.system('setsid python filename.py > /tmp/log1 2>&1 &')