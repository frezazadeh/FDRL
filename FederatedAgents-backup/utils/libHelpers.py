import logging.config


def get_configured_logger(name, level=logging.DEBUG):
    """
    Helper function that provides one customized logger per thread.
    """

    # Configure logging
    logging.basicConfig(format="%(levelname)-10s | %(funcName)s | %(message)s",
                            datefmt='%d/%m/%Y %H:%M:%S')  # %(asctime)-15s |

    logging.addLevelName(logging.ERROR, "\033[1;31m%s\033[1;0m" % logging.getLevelName(logging.ERROR))
    logging.addLevelName(logging.WARNING, "\033[1;33m%s\033[1;0m" % logging.getLevelName(logging.WARNING))
    logging.addLevelName(logging.INFO, "\033[1;34m%s\033[1;0m" % logging.getLevelName(logging.INFO))
    logging.addLevelName(logging.DEBUG, "\033[1;32m%s\033[1;0m" % logging.getLevelName(logging.DEBUG))

    logger = logging.getLogger(name)  # Creates new instance of logger

    if level == "DEBUG":
        level= logging.DEBUG
    if level == "WARNING":
        level =logging.WARNING
    if level == "INFO":
        level = logging.INFO
    if level == "ERROR":
        level = logging.ERROR

    if (len(logger.handlers) == 0):
        FORMAT = "%(name)s.%(levelname)s - %(message)s"
        formatter = logging.Formatter(fmt=FORMAT)
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = 0  # Avoid Messages to propagate till the main handler,i.e. have repetition in the console

    return logger