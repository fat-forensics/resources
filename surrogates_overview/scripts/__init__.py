"""
Surrogate Explainability Overview
=================================

This small library implements a collection of helper functions for a range of
surrogate explainability methods.
Among others, it implements an image classifier based on PyTorch;
a surrogate explainer of image and tabular data; and
a collection of interactive iPyWidgets for no-code experiments with these
explainers.
See <https://github.com/fat-forensics/resources/tree/master/surrogates_overview>
for more details.
"""
# Author: Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

import logging

# Set up logging; enable logging of level INFO and higher
logger = logging.getLogger(__name__)
_logger_handler = logging.StreamHandler()
_logger_formatter = logging.Formatter(
    '%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
    datefmt='%y-%b-%d %H:%M:%S')
_logger_handler.setFormatter(_logger_formatter)
logger.addHandler(_logger_handler)
logger.setLevel(logging.INFO)
