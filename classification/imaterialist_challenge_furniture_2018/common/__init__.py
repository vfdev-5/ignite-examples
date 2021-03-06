import logging


def setup_logger(logger, log_filepath, level=logging.INFO):

    if logger.hasHandlers():
        for h in list(logger.handlers):
            logger.removeHandler(h)

    logger.setLevel(level)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(log_filepath)
    fh.setLevel(level)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(level)
    # create formatter and add it to the handlers
    formatter = logging.Formatter("%(asctime)s|%(name)s|%(levelname)s| %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)


def save_conf(config_file, logdir, logger, writer=None):
    conf_str = """
        Configuration file: {}

        LOG_DIR: {}
            
    """.format(config_file, logdir)
    with open(config_file, 'r') as reader:
        lines = reader.readlines()
        for l in lines:
            conf_str += "\t" + l
    conf_str += "\n\n"
    logger.info(conf_str)
    if writer is not None:
        writer.add_text('Configuration', conf_str)
