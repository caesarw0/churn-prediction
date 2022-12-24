import logging
import logging.config
import datetime as dt
import configparser
import os

def return_logger():
    path = os.getcwd()

    config_filename = path + '\config\logging.conf'
    
    config = configparser.ConfigParser()
    config.read_file(open(config_filename))
    format_from_config = config.get('formatter_simpleFormatter', 'format', raw=True)
    # set log filename
    today = dt.datetime.today()
    log_filename = f"\logs\{today.year}_{today.month:02d}_{today.day:02d}.log"
    log_filepath = os.path.dirname(path) + log_filename
    # read logger conf
    logging.config.fileConfig(config_filename)
    # create logger, setup file handler & add it to logger
    logger = logging.getLogger()
    file_handler = logging.FileHandler(log_filepath)
    file_handler.setFormatter(logging.Formatter(format_from_config))
    logger.addHandler(file_handler)

    return logger