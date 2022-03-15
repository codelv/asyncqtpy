# © 2018 Gerard Marull-Paretas <gerard@teslabs.com>
# © 2014 Mark Harviston <mark.harviston@gmail.com>
# © 2014 Arve Knudsen <arve.knudsen@gmail.com>
# BSD License

"""Mostly irrelevant, but useful utilities common to UNIX and Windows."""
import logging


def with_logger(cls):
    """Class decorator to add a logger to a class."""
    cls_name = cls.__qualname__
    module = cls.__module__
    assert module is not None
    cls_name = f"{module}.{cls_name}"
    cls._logger = logging.getLogger(cls_name)
    return cls
