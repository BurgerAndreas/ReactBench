"""
A computational chemistry package for reaction path calculations
"""

import os
import sys


PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

os.environ["REACTBENCH_PATH"] = PACKAGE_ROOT


from . import Calculators
from . import utils
from . import main_functions
