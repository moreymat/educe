"""This module implements a dumper for the EDU input format

See `<https://github.com/irit-melodi/attelo/blob/master/doc/input.rst>`_
"""

from __future__ import absolute_import, print_function
import codecs
import csv

import six

# FIXME adapt load_edu_input_file to STAC
from educe.annotation import Span  # WIP load_edu_input_file
from educe.corpus import FileId  # WIP load_edu_input_file
from educe.learning.svmlight_format import dump_svmlight_file
from educe.rst_dt.annotation import EDU as RstEDU  # WIP load_edu_input_file

# pylint: disable=invalid-name
# a lot of the names here are chosen deliberately to
# go with sklearn convention


# EDUs
def _dump_edu_input_file(docs, f):
    """Actually do dump"""
    writer = csv.writer(f, dialect=csv.excel_tab)

    for doc in docs:
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


def dump_edu_input_file(docs, f):
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
        _dump_edu_input_file(docs, f)


# FIXME adapt to STAC
def _load_edu_input_file(f, edu_type):
    """Do load."""
    edus = []
    edu2sent = []

    if edu_type == 'rst-dt':
        EDU = RstEDU
    # FIXME support STAC

    reader = csv.reader(f, dialect=csv.excel_tab)
    for line in reader:
        if not line:
            continue
        edu_gid, edu_txt, grouping, subgroup, edu_start, edu_end = line
        # FIXME only works for RST-DT, broken on STAC
        # no subdoc in RST-DT, hence no orig_subdoc in global_id for EDU
        orig_doc, edu_lid = edu_gid.rsplit('_', 1)
        assert grouping == orig_doc  # both are the doc_name
        origin = FileId(orig_doc, None, None, None)
        edu_num = int(edu_lid)
        edu_txt = edu_txt.decode('utf-8')
        edu_start = int(edu_start)
        edu_end = int(edu_end)
        edu_span = Span(edu_start, edu_end)
        edus.append(
            EDU(edu_num, edu_span, edu_txt, origin=origin)
        )
        # edu2sent
        sent_idx = int(subgroup.split('_sent')[1])
        edu2sent.append(sent_idx)
    return {'filename': f.name,
            'edus': edus,
            'edu2sent': edu2sent}


def load_edu_input_file(f, edu_type='rst-dt'):
    """Load a list of EDUs from a file in the EDU input format.

    Parameters
    ----------
    f : str
        Path to the .edu_input file

    edu_type : str, one of {'rst-dt'}
        Type of EDU to load ; 'rst-dt' is the only type currently
        allowed but more should come (unless a unifying type for EDUs
        emerge, rendering this parameter useless).

    Returns
    -------
    data: dict
        Bunch-like object with interesting fields "filename", "edus",
        "edu2sent".
    """
    if edu_type != 'rst-dt':
        raise NotImplementedError(
            "edu_type {} not yet implemented".format(edu_type))
    with codecs.open(f, 'rb', 'utf-8') as f:
        return _load_edu_input_file(f, edu_type)
# end FIXME adapt to STAC


# pairings
def _dump_pairings_file(docs_epairs, f):
    """Actually do dump"""
    writer = csv.writer(f, dialect=csv.excel_tab)

    for epairs in docs_epairs:
        for src, tgt in epairs:
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


def _load_labels_file(f):
    """Actually read the label set from a mapping file.

    Parameters
    ----------
    f : str
        Mapping file, each line pairs an integer index with a label.

    Returns
    -------
    labels : dict from str to int
        Mapping from relation label to integer.
    """
    labels = dict()
    for line in f:
        i, lbl = line.strip().split()
        labels[lbl] = int(i)
    assert labels['__UNK__'] == 0
    return labels


def _load_labels_header(f):
    """Actually read the label set from the header of a features file.

    Previous versions of educe dumped the labels in the header of the
    svmlight features file: The first line was commented and contained
    the list of labels, mapped to indices from 1 to n.

    Parameters
    ----------
    f : str
        Features file, whose first line is a comment with the list of labels.

    Returns
    -------
    labels : dict from str to int
        Mapping from relation label to integer.
    """
    line = f.readline()
    seq = line[1:].split()[1:]
    labels = {lbl: idx for idx, lbl in enumerate(seq, start=1)}
    labels['__UNK__'] = 0
    return labels


def load_labels(f, stored_as='file'):
    """Read label set into a dictionary mapping labels to indices.

    Parameters
    ----------
    f : str
        File containing the labels.
    stored_as : str, one of {'header', 'file'}
        Storage mode of the labelset, as the `header` (commented first
        line) of an svmlight features file, or as an independent `file`
        where each line pairs an integer index with a label.

    Returns
    -------
    labels : dict from str to int
        Mapping from relation label to integer.
    """
    if stored_as == 'header':
        _load_labels = _load_labels_as_header
    elif stored_as == 'file':
        _load_labels = _load_labels_as_file
    else:
        raise ValueError(
            "load_labels: stored_as must be one of {'header', 'file'}")
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


def dump_all(X_gen, y_gen, f, docs, instance_generator, class_mapping=None):
    """Dump a whole dataset: features (in svmlight) and EDU pairs.

    Parameters
    ----------
    X_gen : iterable of iterable of int arrays
        Feature vectors.
    y_gen : iterable of iterable of int
        Ground truth labels.
    f : str
        Output features file path
    docs : list of DocumentPlus
        Documents
    instance_generator : function from doc to iterable of pairs
        TODO
    class_mapping : dict(str, int), optional
        Mapping from label to int. If None, it is ignored so you need
        to check a proper call to dump_labels has been made elsewhere.
        If not None, the list of labels ordered by index is written as
        the header of the svmlight features file, as a comment line.
    """
    # dump EDUs
    edu_input_file = f + '.edu_input'
    dump_edu_input_file(docs, edu_input_file)
    # dump EDU pairings
    pairings_file = f + '.pairings'
    dump_pairings_file((instance_generator(doc) for doc in docs),
                       pairings_file)
    # dump vectorized pairings with label
    # the labelset will be written in a comment at the beginning of the
    # svmlight file
    if class_mapping is not None:
        comment = labels_comment(class_mapping)
    else:
        comment = ''
    dump_svmlight_file(X_gen, y_gen, f, comment=comment)
