"""Dependency format for RST discourse trees.

One line per EDU.
"""

from __future__ import absolute_import, print_function
import codecs
import csv
import os

from educe.rst_dt.corpus import (RELMAP_112_18_FILE, RstRelationConverter,
                                 Reader)
from educe.rst_dt.deptree import RstDepTree
from educe.rst_dt.rst_wsj_corpus import TRAIN_FOLDER, TEST_FOLDER

RELCONV = RstRelationConverter(RELMAP_112_18_FILE).convert_label


def _dump_disdep_file(rst_deptree, f):
    """Actually do dump"""
    writer = csv.writer(f, dialect=csv.excel_tab)

    # 0 is the fake root, there is no point in writing its info
    edus = rst_deptree.edus[1:]
    heads = rst_deptree.heads[1:]
    labels = rst_deptree.labels[1:]
    nucs = rst_deptree.nucs[1:]
    ranks = rst_deptree.ranks[1:]

    for i, (edu, head, label, nuc, rank) in enumerate(
            zip(edus, heads, labels, nucs, ranks), start=1):
        # text of EDU ; some EDUs have newlines in their text, so convert
        # those to simple spaces
        txt = edu.text().replace('\n', ' ')
        clabel = RELCONV(label)
        writer.writerow([i, txt, head, label, clabel, nuc, rank])


def dump_disdep_file(rst_deptree, f):
    """Dump dependency RST tree to a disdep file.

    Parameters
    ----------
    doc: DocumentPlus
        (Rich representation of) the document.
    f: str
        Path of the output file.
    """
    with codecs.open(f, 'wb', 'utf-8') as f:
        _dump_disdep_file(rst_deptree, f)


def dump_disdep_files(rst_deptrees, out_dir):
    """Dump dependency RST trees to a folder.

    This creates one file per RST tree plus a metadata file (encoding of
    n-ary relations, coarse-to-fine mapping for relation labels).
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # metadata
    nary_encs = [x.nary_enc for x in rst_deptrees]
    assert len(set(nary_encs)) == 1
    nary_enc = nary_encs[0]
    f_meta = os.path.join(out_dir, 'metadata')
    with codecs.open(f_meta, mode='w', encoding='utf-8') as f_meta:
        print('nary_enc: {}'.format(nary_enc), file=f_meta)
        print('relmap: {}'.format(RELMAP_112_18_FILE), file=f_meta)

    for rst_deptree in rst_deptrees:
        doc_name = rst_deptree.origin.doc
        f_doc = os.path.join(out_dir, '{}.dis_dep'.format(doc_name))
        dump_disdep_file(rst_deptree, f_doc)


def main():
    """A main that should probably become a proper executable script"""
    # TODO expose these parameters with an argparser
    corpus_dir = os.path.join(
        '/home/mmorey/corpora/rst_discourse_treebank/',
        'data'
    )
    dir_train = os.path.join(corpus_dir, TRAIN_FOLDER)
    dir_test = os.path.join(corpus_dir, TEST_FOLDER)

    out_dir = os.path.join(
        '/home/mmorey/melodi/irit-rst-dt/TMP_disdep_chain_true'
    )
    nary_enc = 'chain'  # 'tree'
    # end TODO

    # convert and dump RST trees from train
    reader_train = Reader(dir_train)
    trees_train = reader_train.slurp()
    dtrees_train = {doc_name: RstDepTree.from_rst_tree(rst_tree,
                                                       nary_enc=nary_enc)
                    for doc_name, rst_tree in trees_train.items()}
    dump_disdep_files(dtrees_train.values(),
                      os.path.join(out_dir, os.path.basename(dir_train)))
    # convert and dump RST trees from test
    reader_test = Reader(dir_test)
    trees_test = reader_test.slurp()
    dtrees_test = {doc_name: RstDepTree.from_rst_tree(rst_tree,
                                                      nary_enc=nary_enc)
                   for doc_name, rst_tree in trees_test.items()}
    dump_disdep_files(dtrees_test.values(),
                      os.path.join(out_dir, os.path.basename(dir_test)))


if __name__ == '__main__':
    main()
