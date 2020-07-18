"""
Logging is very important. It helps you:
- look at old experiments and see what happened
- track down bugs
- monitor ongoing experiments
and many other things.

My current favorite is loguru https://loguru.readthedocs.io/en/stable/index.html
"""
import sys
from loguru import logger


def setup(log_fname=None):
    if log_fname is not None:
        logger.add(log_fname)


def debug(s):
    logger.debug(s)


def info(s):
    logger.info(s)


def warn(s):
    logger.warning(s)
