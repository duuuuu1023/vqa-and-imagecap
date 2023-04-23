import logging
import os
def init_loger(Config):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")

    if not os.path.exists('./log/'):
        os.mkdir('./log/')
    fh = logging.FileHandler('./log/log-' + 'model' +Config.model_name+ '.log', mode='a',
                             encoding='utf-8')
    fh.setFormatter(formatter)

    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s: %(message)s"))
    console.setLevel(logging.INFO)

    logger.addHandler(fh)
    logger.addHandler(console)
    return logger