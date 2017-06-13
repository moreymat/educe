#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Eric Kow
# License: CeCILL-B (French BSD3)

"""
Extract features to CSV files
"""

from __future__ import print_function
import itertools
import os

import educe.corpus
import educe.glozz
import educe.stac
import educe.util

from educe.learning.edu_input_format import (
    dump_all, dump_labels, load_labels)
from educe.learning.vocabulary_format import (
    dump_vocabulary, load_vocabulary)
from educe.rst_dt.corenlp import CoreNlpParser
from educe.rst_dt.corpus import RstDtParser
from educe.rst_dt.learning.args import add_usual_input_args
from educe.rst_dt.learning.doc_vectorizer import (
    DocumentCountVectorizer, DocumentLabelExtractor)
from educe.rst_dt.ptb import PtbParser


NAME = 'extract'


# ----------------------------------------------------------------------
# options
# ----------------------------------------------------------------------

def config_argparser(parser):
    """
    Subcommand flags.
    """
    add_usual_input_args(parser)
    parser.add_argument('corpus', metavar='DIR',
                        help='Corpus dir (eg. data/pilot)')
    # TODO make optional and possibly exclusive from corenlp below
    parser.add_argument('ptb', metavar='DIR',
                        help='PTB directory (eg. PTBIII/parsed/wsj)')
    parser.add_argument('output', metavar='DIR',
                        help='Output directory')
    # add flags --doc, --subdoc, etc to allow user to filter on these things
    educe.util.add_corpus_filters(parser,
                                  fields=['doc'])
    parser.add_argument('--verbose', '-v', action='count',
                        default=1)
    parser.add_argument('--quiet', '-q', action='store_const',
                        const=0,
                        dest='verbose')
    parser.add_argument('--parsing', action='store_true',
                        help='Extract features for parsing')
    parser.add_argument('--vocabulary',
                        metavar='FILE',
                        help='Use given vocabulary for feature output '
                        '(when extracting test data, you may want to '
                        'use the feature vocabulary from the training '
                        'set ')
    # labels
    # TODO restructure ; the aim is to have three options:
    # * fine-grained labelset (no transformation from treebank),
    # * coarse-grained labelset (mapped from fine-grained),
    # * manually specified list of labels (important here is the order
    # of the labels, that implicitly maps labels as strings to integers)
    # ... but what to do is not 100% clear right now
    parser.add_argument('--labels',
                        metavar='FILE',
                        help='Read label set from given feature file '
                        '(important when extracting test data)')
    parser.add_argument('--coarse',
                        action='store_true',
                        help='use coarse-grained labels')
    parser.add_argument('--fix_pseudo_rels',
                        action='store_true',
                        help='fix pseudo-relation labels')
    # NEW use CoreNLP's output for tokenization and syntax (+coref?)
    parser.add_argument('--corenlp_out_dir', metavar='DIR',
                        help='CoreNLP output directory')
    # end NEW
    # NEW lecsie features
    parser.add_argument('--lecsie_data_dir', metavar='DIR',
                        help='LECSIE features directory')
    # end NEW

    parser.add_argument('--debug', action='store_true',
                        help='Emit fields used for debugging purposes')
    parser.add_argument('--experimental', action='store_true',
                        help='Enable experimental features '
                             '(currently none)')
    # 2016-09-12 nary_enc: chain vs tree transform
    parser.add_argument('--nary_enc', default='chain',
                        choices=['chain', 'tree'],
                        help='Encoding for n-ary nodes')
    # end nary_enc
    # WIP 2016-07-15 same-unit
    parser.add_argument('--instances',
                        choices=['edu-pairs', 'same-unit'],
                        default='edu-pairs',
                        help="Selection of instances")
    # end WIP same-unit
    # 2016-09-30 enable to choose between unordered and ordered pairs
    parser.add_argument('--unordered_pairs',
                        action='store_true',
                        help=("Instances are unordered pairs: "
                              "(src, tgt) == (tgt, src)"))
    # WIP 2017-02-02 toggle between corpus- and doc-centric feature
    # extraction
    parser.add_argument('--file_split',
                        choices=['doc', 'corpus'],
                        default='corpus',
                        help=("Level of granularity for each set of"
                              "files: 'doc' produces one set of files per"
                              "document ; 'corpus' one set of files per"
                              "corpus split (e.g. 'train', 'test')"))
    # end WIP toggle between corpus- and doc-centric feature extraction
    parser.set_defaults(func=main)


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------
def extract_dump_instances(docs, instance_generator, feature_set,
                           lecsie_data_dir, vocabulary, split_feat_space,
                           labels, live, ordered_pairs, output, corpus,
                           file_split='corpus'):
    """Extract and dump instances.

    Parameters
    ----------
    docs : list of DocumentPlus
        Documents
    instance_generator : (string, function)
        Instance generator: the first element is a string descriptor of
        the instance generator, the second is the instance generator
        itself: a function from DocumentPlus to list of EDU pairs.
    vocabulary : filepath
        Path to vocabulary
    split_feat_space : string
        Splitter for feature space
    labels : filepath?
        Path to labelset?
    live : TODO
        TODO
    ordered_pairs : boolean
        If True, DU pairs (instances) are ordered pairs, i.e.
        (src, tgt) <> (tgt, src).
    output : string
        Path to the output directory, e.g. 'TMP/data'.
    corpus : TODO
        TODO
    """
    # get instance generator and its descriptor
    instance_descr, instance_gen = instance_generator
    # setup persistency
    if not os.path.exists(output):
        os.makedirs(output)
    corpus_name = os.path.basename(corpus)

    if live:
        fn_out = 'extracted-features.{instance_descr}'.format(
            instance_descr=instance_descr)
    else:
        fn_out = '{corpus_name}.relations.{instance_descr}'.format(
            corpus_name=corpus_name,
            instance_descr=instance_descr)

    # vocabulary, labels
    fn_ext = '.sparse'  # our extension for sparse datasets (svmlight)
    vocab_file = os.path.join(output, fn_out + fn_ext + '.vocab')
    labels_file = os.path.join(output, fn_out + '.labels')
    # WIP 2016-08-29 output folder, will contain n files per doc
    # ex: TRAINING/wsj_0601.out.relations.all-pairs.sparse
    out_dir = os.path.join(output, os.path.basename(corpus))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # end WIP output folder

    # extract vectorized samples
    if vocabulary is not None:
        vocab = load_vocabulary(vocabulary)
        min_df = 1
    else:
        vocab = None
        min_df = 5
    vzer = DocumentCountVectorizer(instance_gen, feature_set,
                                   lecsie_data_dir=lecsie_data_dir,
                                   min_df=min_df, vocabulary=vocab,
                                   split_feat_space=split_feat_space)
    # pylint: disable=invalid-name
    # X, y follow the naming convention in sklearn
    if vocabulary is not None:
        X_gen = vzer.transform(docs)
    else:
        X_gen = vzer.fit_transform(docs)
    # pylint: enable=invalid-name

    # extract class label for each instance
    if live:
        y_gen = itertools.repeat(0)
    else:
        if labels is not None:
            labelset = load_labels(labels, stored_as='file')
        else:
            labelset = None
        labtor = DocumentLabelExtractor(
            instance_gen, ordered_pairs=ordered_pairs, labelset=labelset)
        if labels is not None:
            labtor.fit(docs)
            y_gen = labtor.transform(docs)
        else:
            # y_gen = labtor.fit_transform(rst_corpus)
            # fit then transform enables to get classes_ for the dump
            labtor.fit(docs)
            y_gen = labtor.transform(docs)

    # dump instances to files
    if file_split == 'doc':
        # one set of files per document
        for doc, X, y in itertools.izip(docs, X_gen, y_gen):
            # dump EDUs and features in svmlight format
            doc_name = doc.key.doc
            # TODO refactor
            if live:
                fn_out = 'extracted-features.{}{}'.format(
                    instance_descr, fn_ext)
            else:
                fn_out = '{}.relations.{}{}'.format(
                    doc_name, instance_descr, fn_ext)
            out_file = os.path.join(out_dir, fn_out)
            # end TODO refactor
            dump_all([X], [y], out_file, [doc], instance_gen)
    elif file_split == 'corpus':
        # one set of files per corpus (in fact, corpus split)
        if live:
            fn_out = 'extracted-features.{instance_descr}{fn_ext}'.format(
                instance_descr=instance_descr, fn_ext=fn_ext)
        else:
            fn_out = ('{corpus_name}.relations.{instance_descr}{fn_ext}'
                      .format(
                          corpus_name=corpus_name,
                          instance_descr=instance_descr,
                          fn_ext=fn_ext))
        out_file = os.path.join(out_dir, fn_out)
        dump_all(X_gen, y_gen, out_file, docs, instance_gen)
    else:
        raise ValueError('Unknown value for args.file_split : {}'.format(
            args.file_split))

    # dump labelset
    if labels is not None:
        # relative path to get a correct symlink
        existing_labels = os.path.relpath(
            labels, start=os.path.dirname(labels_file))
        # c/c from attelo.harness.util.force_symlink()
        if os.path.islink(labels_file):
            os.unlink(labels_file)
        elif os.path.exists(labels_file):
            oops = ("Can't force symlink from " + labels +
                    " to " + labels_file +
                    " because a file of that name already exists")
            raise ValueError(oops)
        os.symlink(existing_labels, labels_file)
        # end c/c
    else:
        dump_labels(labtor.labelset_, labels_file)

    # dump vocabulary
    if vocabulary is not None:
        # FIXME relative path to get a correct symlink
        existing_vocab = os.path.relpath(
            vocabulary, start=os.path.dirname(vocab_file))
        # c/c from attelo.harness.util.force_symlink()
        if os.path.islink(vocab_file):
            os.unlink(vocab_file)
        elif os.path.exists(vocab_file):
            oops = ("Can't force symlink from " + vocabulary +
                    " to " + vocab_file +
                    " because a file of that name already exists")
            raise ValueError(oops)
        os.symlink(existing_vocab, vocab_file)
        # end c/c
    else:
        dump_vocabulary(vzer.vocabulary_, vocab_file)


def main(args):
    "main for feature extraction mode"
    # retrieve parameters
    feature_set = args.feature_set
    live = args.parsing
    lecsie_data_dir = args.lecsie_data_dir  # NEW lecsie features
    # RST data
    # fileX docs are currently not supported by CoreNLP
    exclude_file_docs = args.corenlp_out_dir

    rst_reader = RstDtParser(args.corpus, args, coarse_rels=args.coarse,
                             fix_pseudo_rels=args.fix_pseudo_rels,
                             nary_enc=args.nary_enc,
                             exclude_file_docs=exclude_file_docs)
    rst_corpus = rst_reader.corpus
    # TODO: change educe.corpus.Reader.slurp*() so that they return an object
    # which contains a *list* of FileIds and a *list* of annotations
    # (see sklearn's Bunch)
    # on creation of these lists, one can impose the list of names to be
    # sorted so that the order in which docs are iterated is guaranteed
    # to be always the same

    # syntactic preprocessing
    if args.corenlp_out_dir:
        # get the precise path to CoreNLP parses for the corpus currently used
        # the folder layout of CoreNLP's output currently follows that of the
        # corpus: RSTtrees-main-1.0/{TRAINING,TEST}, RSTtrees-double-1.0
        # FIXME clean rewrite ; this could mean better modelling of the corpus
        # subparts/versions, e.g. RST corpus have "version: 1.0", annotators
        # "main" or "double"

        # find the suffix of the path name that starts with RSTtrees-*
        # FIXME find a cleaner way to do this ;
        # should probably use pathlib, included in the standard lib
        # for python >= 3.4
        try:
            rel_idx = (args.corpus).index('RSTtrees-WSJ-')
        except ValueError:
            # if no part of the path starts with "RSTtrees", keep the
            # entire path (no idea whether this is good)
            relative_corpus_path = args.corpus
        else:
            relative_corpus_path = args.corpus[rel_idx:]

        corenlp_out_dir = os.path.join(args.corenlp_out_dir,
                                       relative_corpus_path)
        csyn_parser = CoreNlpParser(corenlp_out_dir)
    else:
        # TODO improve switch between gold and predicted syntax
        # PTB data
        csyn_parser = PtbParser(args.ptb)
    # FIXME
    print('offline syntactic preprocessing: ready')

    # align EDUs with sentences, tokens and trees from PTB
    def open_plus(doc):
        """Open and fully load a document.

        Parameters
        ----------
        doc : educe.corpus.FileId
            Document key.

        Returns
        -------
        doc : DocumentPlus
            Rich representation of the document.
        """
        # create a DocumentPlus
        doc = rst_reader.decode(doc)
        # populate it with layers of info
        # tokens
        doc = csyn_parser.tokenize(doc)
        # syn parses
        doc = csyn_parser.parse(doc)
        # disc segments
        doc = rst_reader.segment(doc)
        # disc parse
        doc = rst_reader.parse(doc)
        # pre-compute the relevant info for each EDU
        doc = doc.align_with_doc_structure()
        # logical order is align with tokens, then align with trees
        # but aligning with trees first for the PTB enables
        # to get proper sentence segmentation
        doc = doc.align_with_trees()
        doc = doc.align_with_tokens()
        # dummy, fallback tokenization if there is no PTB gold or silver
        doc = doc.align_with_raw_words()

        return doc

    # generate DocumentPluses
    # TODO remove sorted() once educe.corpus.Reader is able
    # to iterate over a stable (sorted) list of FileIds
    docs = [open_plus(doc) for doc in sorted(rst_corpus)]
    # instance generator
    ordered_pairs = not args.unordered_pairs  # 2016-09-30
    if args.instances == 'same-unit':
        # WIP 2016-07-08 pre-process to find same-units
        instance_generator = ('same-unit',
                              lambda doc: doc.same_unit_candidates())
        split_feat_space = None
    elif args.instances == 'edu-pairs':
        # all pairs of EDUs
        instance_generator = ('edu-pairs',
                              lambda doc: doc.all_edu_pairs(
                                  ordered=ordered_pairs))
        split_feat_space = 'dir_sent'
    # do the extraction
    extract_dump_instances(docs, instance_generator, feature_set,
                           lecsie_data_dir, args.vocabulary, split_feat_space,
                           args.labels, live, ordered_pairs, args.output,
                           args.corpus, file_split=args.file_split)
