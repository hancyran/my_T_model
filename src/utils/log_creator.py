import os
import logging
import time


def create_logger(output_path, comment):
    # archive_name = "{}_{}.tgz".format(cfg_name, time.strftime('%Y-%m-%d-%H-%M'))
    # archive_path = os.path.join(os.path.join(final_output_path, archive_name))
    # pack_experiment(".",archive_path)

    date = time.strftime('%m-%d %H:%M', time.localtime(time.time() + 3600 * 8))

    log_file = '{}_{}.log'.format(date, comment)

    head = '%(levelname)s %(asctime)-15s %(message)s'
    logging.basicConfig(filename=os.path.join(output_path, log_file), format=head)
    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)

    return logger
