# Author: Caesar Wong
# Date: 2022-12-24

"""
A script that read the logging configuration file and return a logger for other function.
The logger has a file handler which uses the script running date as the file name.

Usage: 
(in another py script)
    from utils.get_logger import return_logger
    logger = return_logger()

    logger.info("sample logging info msg")

"""

import logging
import logging.config
import datetime as dt
import configparser
import os

def return_logger():
    '''
    return a logger that is ready to use
    
    Parameters
    ----------
    
    Description
    ----------
    1. Get the logging config file
    2. Generate filename (today's date, yyyy-mm-dd.log)
    3. Configure the logger with config file
    4. Add file handler to the logger (with the generated filename)
    5. Add formatter to the file handler
    6. Return logger
    '''

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