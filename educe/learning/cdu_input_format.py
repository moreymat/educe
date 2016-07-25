"""This module implements a dumper for the CDU input format.

This format is WIP.
"""

from __future__ import absolute_import, print_function
import codecs
import csv

import six

from .edu_input_format import labels_comment
from .svmlight_format import dump_svmlight_file

# pylint: disable=invalid-name
# a lot of the names here are chosen deliberately to
# go with scikit convention

# EDUs
def _dump_cdu_input_file(doc_cdus, f):
    """Actually do dump"""
    writer = csv.writer(f, dialect=csv.excel_tab)

    for cdus in doc_cdus:
        for cdu in cdus:
            # cdu: (cdu_identifier, tuple(du_identifiers))
            writer.writerow([cdu[0]] + list(cdu[1]))


def dump_cdu_input_file(doc_cdus, f):
    """Dump a dataset in the CDU input format.

    Each element of doc_cdus is a list of CDUs for a document.
    Each CDU is described by its identifier followed by the list of
    identifiers of its members.
    """
    with open(f, 'wb') as f:
        _dump_cdu_input_file(doc_cdus, f)


# pairings
def _dump_cdu_pairings_file(docs_pairs, f):
    """Actually do dump"""
    writer = csv.writer(f, dialect=csv.excel_tab)

    for du_pairs in docs_pairs:
        for src, tgt in du_pairs:
            src_id = src[0] if isinstance(src, tuple) else src.identifier()
            tgt_id = tgt[0] if isinstance(tgt, tuple) else tgt.identifier()
            writer.writerow([src_id, tgt_id])


def dump_cdu_pairings_file(epairs, f):
    """Dump the DU pairings (with at least one CDU)"""
    with open(f, 'wb') as f:
        _dump_cdu_pairings_file(epairs, f)


def dump_all_cdus(X_gen, y_gen, f, class_mapping, docs, doc_cdus,
                  instance_generator):
    """Dump instances from/to CDUs: features (in svmlight) and DU pairs.

    Parameters
    ----------
    X_gen: iterable of int arrays
        TODO
    y_gen: iterable of int
        TODO
    f: TODO
        Output features file path
    class_mapping: dict(string, int)
        Mapping from label to int
    docs: iterable of DocumentPlus
        Documents
    doc_cdus: iterable of iterable of CDUs
        List of CDUs for each document
    instance_generator: function from DocumentPlus to iterable of pairs
        Function that generates an iterable of pairs from a
        DocumentPlus.
    """
    # dump: EDUs, pairings, vectorized pairings with label
    cdu_input_file = f + '.cdu_input'
    dump_cdu_input_file(doc_cdus, cdu_input_file)

    pairings_file = f + '.cdu_pairings'
    dump_cdu_pairings_file((instance_generator(doc) for doc in docs),
                           pairings_file)

    # the labelset will be written in a comment at the beginning of the
    # svmlight file
    comment = labels_comment(class_mapping)
    dump_svmlight_file(X_gen, y_gen, f, comment=comment)
