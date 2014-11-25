#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Eric Kow
# License: CeCILL-B (French BSD3)

"""
Emit a list of known features
"""

from __future__ import print_function

from ..base import read_help_inputs
from .. import features_li2014 as feature_set  # TODO turn into cmd line arg

NAME = 'features'

# ----------------------------------------------------------------------
# options
# ----------------------------------------------------------------------


def config_argparser(parser):
    """
    Subcommand flags.
    """
    parser.set_defaults(func=main)

# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------


def main(args):
    "main for feature listing mode"
    inputs = read_help_inputs(args)
    print(feature_set.PairKeys(inputs).help_text())
