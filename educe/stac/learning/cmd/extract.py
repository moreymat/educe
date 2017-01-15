#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Eric Kow
# License: BSD3

"""
Extract features to CSV files
"""

from __future__ import print_function
from os import path as fp
import itertools
import os
import sys

from educe.learning.keygroup_vectorizer import (KeyGroupVectorizer)
from educe.stac.annotation import (DIALOGUE_ACTS,
                                   SUBORDINATING_RELATIONS,
                                   COORDINATING_RELATIONS)
from educe.stac.learning.doc_vectorizer import (
    DialogueActVectorizer, LabelVectorizer, mk_high_level_dialogues,
    extract_pair_features, extract_single_features, read_corpus_inputs)


import educe.corpus
from educe.learning.edu_input_format import (dump_all,
                                             labels_comment,
                                             dump_svmlight_file,
                                             dump_edu_input_file)
from educe.learning.vocabulary_format import (dump_vocabulary,
                                              load_vocabulary)
import educe.glozz
import educe.stac
import educe.util


NAME = 'extract'


# ----------------------------------------------------------------------
# options
# ----------------------------------------------------------------------


def config_argparser(parser):
    """
    Subcommand flags.
    """
    parser.add_argument('corpus', metavar='DIR',
                        help='Corpus dir (eg. data/pilot)')
    parser.add_argument('resources', metavar='DIR',
                        help='Resource dir (eg. data/resource)')
    parser.add_argument('output', metavar='DIR',
                        help='Output directory')
    # add flags --doc, --subdoc, etc to allow user to filter on these things
    educe.util.add_corpus_filters(parser,
                                  fields=['doc', 'subdoc', 'annotator'])
    parser.add_argument('--verbose', '-v', action='count',
                        default=1)
    parser.add_argument('--quiet', '-q', action='store_const',
                        const=0,
                        dest='verbose')
    parser.add_argument('--single', action='store_true',
                        help="Features for single EDUs (instead of pairs)")
    parser.add_argument('--parsing', action='store_true',
                        help='Extract features for parsing')
    parser.add_argument('--vocabulary',
                        metavar='FILE',
                        help='Vocabulary file (for --parsing mode)')
    parser.add_argument('--ignore-cdus', action='store_true',
                        help='Avoid going into CDUs')
    parser.add_argument('--strip-mode',
                        choices=['head', 'broadcast', 'custom'],
                        default='head',
                        help='CDUs stripping method (if going into CDUs)')
    parser.set_defaults(func=main)

# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------


def main_single(args):
    """Extract feature vectors for single EDUs in the corpus."""
    inputs = read_corpus_inputs(args)
    stage = 'unannotated' if args.parsing else 'units'
    dialogues = list(mk_high_level_dialogues(inputs, stage))
    instance_generator = lambda x: x.edus[1:]  # drop fake root

    # pylint: disable=invalid-name
    # scikit-convention
    feats = extract_single_features(inputs, stage)
    vzer = KeyGroupVectorizer()
    # TODO? just transform() if args.parsing or args.vocabulary?
    X_gen = vzer.fit_transform(feats)
    # pylint: enable=invalid-name
    labtor = DialogueActVectorizer(instance_generator, DIALOGUE_ACTS)
    y_gen = labtor.transform(dialogues)

    # create directory structure: {output}/{corpus}/
    outdir = args.output
    corpus_name = fp.basename(args.corpus)
    outdir_corpus = fp.join(outdir, corpus_name)
    if not fp.exists(outdir_corpus):
        os.makedirs(outdir_corpus)

    # list dialogue acts
    comment = labels_comment(labtor.labelset_)

    # dump: EDUs, pairings, vectorized pairings with label
    # WIP switch to a document (here dialogue) centric generation of data
    for dia, X, y in itertools.izip(dialogues, X_gen, y_gen):
        dia_id = dia.grouping
        print('dump dialogue', dia_id)
        # these paths should go away once we switch to a proper dumper
        feat_file = fp.join(outdir_corpus,
                            '{dia_id}.dialogue-acts.sparse'.format(
                                dia_id=dia_id))
        edu_input_file = '{feat_file}.edu_input'.format(feat_file=feat_file)
        dump_edu_input_file(dia, edu_input_file)
        dump_svmlight_file(X, y, feat_file, comment=comment)
    # end WIP

    # dump vocabulary
    # WIP 2017-01-11 we might need to insert ".{instance_descr}",
    # with e.g. instance_descr='edus', before ".sparse", so as to match
    # the naming scheme currently used for RST
    vocab_file = fp.join(outdir,
                         '{corpus_name}.dialogue-acts.sparse.vocab'.format(
                             corpus_name=corpus_name))
    dump_vocabulary(vzer.vocabulary_, vocab_file)


def main_pairs(args):
    """Extract feature vectors for pairs of EDUs in the corpus."""
    inputs = read_corpus_inputs(args)
    stage = 'units' if args.parsing else 'discourse'
    dialogues = list(mk_high_level_dialogues(inputs, stage))
    instance_generator = lambda x: x.edu_pairs()

    labels = frozenset(SUBORDINATING_RELATIONS +
                       COORDINATING_RELATIONS)

    # pylint: disable=invalid-name
    # X, y follow the naming convention in sklearn
    feats = extract_pair_features(inputs, stage)
    vzer = KeyGroupVectorizer()
    if args.parsing or args.vocabulary:
        vzer.vocabulary_ = load_vocabulary(args.vocabulary)
        X_gen = vzer.transform(feats)
    else:
        X_gen = vzer.fit_transform(feats)
    # pylint: enable=invalid-name
    labtor = LabelVectorizer(instance_generator, labels,
                             zero=args.parsing)
    y_gen = labtor.transform(dialogues)

    # create directory structure
    outdir = args.output
    corpus_name = fp.basename(args.corpus)
    outdir_corpus = fp.join(outdir, corpus_name)
    if not fp.exists(outdir_corpus):
        os.makedirs(outdir_corpus)

    # WIP switch to a document (here dialogue) centric generation of data
    for dia, X, y in itertools.izip(dialogues, X_gen, y_gen):
        dia_id = dia.grouping
        # these paths should go away once we switch to a proper dumper
        out_file = fp.join(outdir_corpus,
                           '{dia_id}.relations.sparse'.format(
                               dia_id=dia_id))
        dump_all(X, y, out_file, dia, instance_generator)
    # end WIP

    # dump vocabulary
    vocab_file = fp.join(outdir,
                         '{corpus_name}.relations.sparse.vocab'.format(
                             corpus_name=corpus_name))
    dump_vocabulary(vzer.vocabulary_, vocab_file)


def main(args):
    "main for feature extraction mode"

    if args.parsing and not args.vocabulary:
        sys.exit("Need --vocabulary if --parsing is enabled")
    if args.parsing and args.single:
        sys.exit("Can't mixing --parsing and --single")
    elif args.single:
        main_single(args)
    else:
        # main_pairs(args)  # DEBUG commented
        pass  # DEBUG
