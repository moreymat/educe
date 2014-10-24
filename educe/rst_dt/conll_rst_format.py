#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Authors: Mathieu Morey
# License: CeCILL-B (BSD3-like)

"""This module implements a loader and dumper for the CoNLL-RST format.

The CoNLL-RST format is a self-contained CoNLL-style format for RST discourse
trees.
"""

import nltk.tree

from educe.external.parser import SearchableTree


_FORMAT = ('doc_id ',
           'part_num ',
           'word_num ',
           'word ',
           'pos ',
           'parse_bit ',
           'lemma ',
           'pred_frameset_id ',
           'word_sense ',
           'discourse_bit'
)


def parse_bits(ptree):
    """Get the parse bit for each leaf of a tree."""
    def _parse_bits(ptree, prefix=[], suffix=[]):
        """Internal recursive generator"""
        if isinstance(ptree, SearchableTree):
            # empty prefix for all kids but 1st
            prefixes = [[] for kid in ptree]
            prefixes[0] = prefix
            prefixes[0].append('({lbl}'.format(lbl=ptree.label()))
            # empty suffix for all kids but last
            suffixes = [[] for kid in ptree]
            suffixes[-1] = suffix
            suffixes[-1].append(')')
            # recursive calls
            for kid_id, kid in enumerate(ptree):
                for pbit in _parse_bits(kid, prefixes[kid_id], suffixes[kid_id]):
                    yield pbit
        else:  # terminal node (token)
            # do not output pre-terminal: prefix[:-1], suffix[:-1]
            pbit = '{p}*{s}'.format(p=''.join(prefix[:-1]),
                                    s=''.join(reversed(suffix[:-1])))
            yield pbit
    # return a list, but we could return the generator itself
    return list(pbit for pbit in _parse_bits(ptree))


def _dump_conll_rst(doc_id, rst_tree, ptb_trees, f):
    """First attempt at doing the dump."""
    for ptree in ptb_trees:
        conll_toks = []
        pbits = parse_bits(ptree)

        for token_num, token in enumerate(ptree.leaves()):
            parse_bit = pbits[token_num]
            disc_bit = '_'  # TODO: discourse bits
            s = [doc_id.doc,
                 doc_id.subdoc,
                 token_num,
                 token.word,
                 token.tag,
                 parse_bit,
                 '_',  # lemma
                 '_',  # pred_frameset_id
                 '_',  # word_sense
                 disc_bit
                 ]
            conll_toks.append(s)
            # f.write(s)
            print '\t'.join(str(field) for field in s)  # DEBUG


def dump_conll_rst(doc_id, rst_tree, ptb_trees, f):
    """Dump a discourse annotated document to a CoNLL-RST formatted file."""
    _dump_conll_rst(doc_id, rst_tree, ptb_trees, f)


#
# TESTS
#
def flat_str_parse_bits(conll_toks):
    """Get a (flat string) parse tree from a list of CoNLL tokens."""
    result = []
    for token in conll_toks:
        s = token[5].replace('*',
                             '({tag} {word})'.format(tag=token[4],
                                                     word=token[3]))
        result.append(s)
    return ''.join(result)

def flat_str_tree(tree):
    """Get a flat string representation of a tree."""
    if isinstance(tree, nltk.tree.Tree):
        return '({lbl}{children})'.format(lbl=tree.label(),
                                          children=''.join(map(flat_str_tree, tree)))
    else:
        return ' {word}'.format(word=tree.word)

def _test_parse_bits(ptree, conll_toks):
    """Test method for parse bits generator.
    TODO: make it clean and usable, maybe move elsewhere.
    """
    str_ptree = flat_str_tree(ptree)  # flat string of the PTB tree
    str_parse_bits = flat_str_parse_bits(conll_toks)
    try:
        assert(str_ptree == str_parse_bits)
    except AssertionError as ae:
        print(str_ptree)
        print(str_parse_bits)
        print(pbits)
        print(ae)
    
