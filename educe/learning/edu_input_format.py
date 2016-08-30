"""This module implements a dumper for the EDU input format

See `<https://github.com/irit-melodi/attelo/blob/master/doc/input.rst>`_
"""

from __future__ import absolute_import, print_function
import codecs
import csv

import six

from .svmlight_format import dump_svmlight_file

# pylint: disable=invalid-name
# a lot of the names here are chosen deliberately to
# go with sklearn convention

# EDUs
def _dump_edu_input_file(doc, f):
    """Actually do dump"""
    writer = csv.writer(f, dialect=csv.excel_tab)

    edus = doc.edus
    grouping = doc.grouping
    edu2sent = doc.edu2sent
    assert edus[0].is_left_padding()
    for i, edu in enumerate(edus[1:], start=1):  # skip the fake root
        edu_gid = edu.identifier()
        # some EDUs have newlines in their text (...):
        # convert to spaces
        edu_txt = edu.text().replace('\n', ' ')
        # subgroup: sentence identifier, backoff on EDU id
        sent_idx = edu2sent[i]
        if sent_idx is None:
            subgroup = edu_gid
        elif isinstance(sent_idx, six.string_types):
            subgroup = sent_idx
        else:
            subgroup = '{}_sent{}'.format(grouping, sent_idx)
        edu_start = edu.span.char_start
        edu_end = edu.span.char_end
        writer.writerow([edu_gid,
                         edu_txt.encode('utf-8'),
                         grouping,
                         subgroup,
                         edu_start,
                         edu_end])


def dump_edu_input_file(doc, f):
    """Dump a dataset in the EDU input format.

    Each document must have:

    * edus: sequence of edu objects
    * grouping: string (some sort of document id)
    * edu2sent: int -> int or string or None (edu num to sentence num)

    The EDUs must provide:

    * identifier(): string
    * text(): string

    """
    with open(f, 'wb') as f:
        _dump_edu_input_file(doc, f)


# pairings
def _dump_pairings_file(doc_epairs, f):
    """Actually do dump"""
    writer = csv.writer(f, dialect=csv.excel_tab)

    for src, tgt in doc_epairs:
        src_gid = src.identifier()
        tgt_gid = tgt.identifier()
        writer.writerow([src_gid, tgt_gid])


def dump_pairings_file(epairs, f):
    """Dump the EDU pairings"""
    with open(f, 'wb') as f:
        _dump_pairings_file(epairs, f)


def labels_comment(class_mapping):
    """Return a string listing class labels in the format that
    attelo expects
    """
    classes_ = [lbl for lbl, _ in sorted(class_mapping.items(),
                                         key=lambda x: x[1])]
    # first item should be reserved for unknown labels
    # we don't want to output this
    classes_ = classes_[1:]
    if classes_:
        comment = 'labels: {}'.format(' '.join(classes_))
    else:
        comment = None
    return comment


def _load_labels(f):
    """Actually read the label set"""
    labels = dict()
    for line in f:
        i, lbl = line.strip().split()
        labels[lbl] = int(i)
    assert labels['__UNK__'] == 0
    return labels


def load_labels(f):
    """Read label set into a dictionary mapping labels to indices"""
    with codecs.open(f, 'r', 'utf-8') as f:
        return _load_labels(f)


def _dump_labels(labelset, f):
    """Do dump labels"""
    for lbl, i in sorted(labelset.items(), key=lambda x: x[1]):
        f.write('{}\t{}\n'.format(i, lbl))


def dump_labels(labelset, f):
    """Dump labelset as a mapping from label to index.

    Parameters
    ----------
    labelset: dict(label, int)
        Mapping from label to index.
    """
    with codecs.open(f, 'wb', 'utf-8') as f:
        _dump_labels(labelset, f)


def dump_all(X, y, f, class_mapping, doc, instance_generator):
    """Dump a whole dataset: features (in svmlight) and EDU pairs.

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
    instance_generator: function from DocumentPlus to iterable of pairs
        Function that generates an iterable of pairs from a
        DocumentPlus.
    """
    # dump EDUs
    edu_input_file = f + '.edu_input'
    dump_edu_input_file(doc, edu_input_file)
    # dump EDU pairings
    pairings_file = f + '.pairings'
    dump_pairings_file(instance_generator(doc), pairings_file)
    # dump vectorized pairings with label
    dump_svmlight_file(X, y, f)
