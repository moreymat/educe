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
# go with sklearn convention

# EDUs
def _dump_cdu_input_file(doc_cdus, f):
    """Actually do dump"""
    writer = csv.writer(f, dialect=csv.excel_tab)

    for cdu in doc_cdus:
        # cdu: (cdu_identifier, tuple(du_identifiers))
        writer.writerow([cdu[0]] + list(cdu[1]))


def dump_cdu_input_file(doc_cdus, f):
    """Dump a dataset in the CDU input format.

    Parameters
    ----------
    doc_cdus: list of CDUs
        CDUs for the document ; Each CDU is described by its identifier
        followed by the list of identifiers of its members.
    f: filepath
        File for the dump.
    """
    with open(f, 'wb') as f:
        _dump_cdu_input_file(doc_cdus, f)


# pairings
def _dump_cdu_pairings_file(doc_pairs, f):
    """Actually do dump"""
    writer = csv.writer(f, dialect=csv.excel_tab)

    for src, tgt in doc_pairs:
        src_id = src[0] if isinstance(src, tuple) else src.identifier()
        tgt_id = tgt[0] if isinstance(tgt, tuple) else tgt.identifier()
        writer.writerow([src_id, tgt_id])


def dump_cdu_pairings_file(epairs, f):
    """Dump the DU pairings (with at least one CDU)"""
    with open(f, 'wb') as f:
        _dump_cdu_pairings_file(epairs, f)


def dump_all_cdus(X, y, f, class_mapping, doc, doc_cdus,
                  instance_generator):
    """Dump instances from/to CDUs: features (in svmlight) and DU pairs.

    Parameters
    ----------
    X: iterable of int arrays
        TODO
    y: iterable of int
        TODO
    f: TODO
        Output features file path
    class_mapping: dict(string, int)
        Mapping from label to int
    doc: DocumentPlus
        Document
    doc_cdus: iterable of iterable of CDUs
        List of CDUs for each document
    instance_generator: function from DocumentPlus to iterable of pairs
        Function that generates an iterable of pairs from a
        DocumentPlus.
    """
    # dump CDUs
    cdu_input_file = f + '.cdu_input'
    dump_cdu_input_file(doc_cdus, cdu_input_file)
    # dump pairings supported by CDUs
    pairings_file = f + '.cdu_pairings'
    dump_cdu_pairings_file(instance_generator(doc), pairings_file)
    # dump vectorized pairings with label
    dump_svmlight_file(X, y, f)
