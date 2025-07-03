# © 2018 Gerard Marull-Paretas <gerard@teslabs.com>
# © 2014 Mark Harviston <mark.harviston@gmail.com>
# © 2014 Arve Knudsen <arve.knudsen@gmail.com>
# BSD License
import os
import sys
import logging

from pytest import fixture

log_fmt = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
logging.basicConfig(level=logging.DEBUG, format=log_fmt)

# Ensure examples are in sys path 
sys.path.append(os.path.dirname(os.path.dirname(__file__))) 

@fixture(scope="session")
def application():
    from asyncqtpy import QApplication

    return QApplication([])
