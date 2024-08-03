# -*- coding: utf-8 -*-

"""
Optimization  toolkit
=========================================
NNO is an optimization  toolkit that enables
researchers to test variants of the DFO and PSO technique in different contexts.
Users can define their own function, or use one of the benchmark functions
in the library. It is built on top of :code:`numpy` and :code:`scipy`, and
is very extensible to accommodate other PSO variations.
"""

__author__ = """Kalong Boniface"""
__email__ = "kalongboniface79@gmail.com"
__version__ = "0"

from .single import global_best, local_best, general_optimizer
from .discrete import binary
from .utils.decorators import cost

__all__ = ["global_best", "local_best", "general_optimizer", "binary", "cost"]