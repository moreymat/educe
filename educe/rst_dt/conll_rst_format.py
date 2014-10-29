#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Authors: Mathieu Morey
# License: CeCILL-B (BSD3-like)

"""This module implements a loader and dumper for the CoNLL-RST format.

The CoNLL-RST format is a self-contained CoNLL-style format for RST discourse
trees.
"""

import itertools
import re
from collections import defaultdict

import nltk.tree

import educe.corpus
from educe.external.parser import SearchableTree
from .parse import parse_rst_dt_tree
from ..internalutil import treenode


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


def discourse_bits(rtree):
    """Get the discourse bit for each leaf of an RST tree."""
    def _discourse_bits(rtree, prefix=[], suffix=[]):
        """Internal recursive generator"""
        if isinstance(rtree, SearchableTree):
            node = treenode(rtree)
            # empty prefix for all kids but 1st
            prefixes = [[] for kid in rtree]
            prefixes[0] = prefix
            prefixes[0].append('({nuc}:{rel}'.format(nuc=node.nuclearity[0],
                                                     rel=node.rel))
            # empty suffix for all kids but last
            suffixes = [[] for kid in rtree]
            suffixes[-1] = suffix
            suffixes[-1].append(')')
            # recursive calls
            for kid_id, kid in enumerate(rtree):
                for dbit in _discourse_bits(kid, prefixes[kid_id],
                                            suffixes[kid_id]):
                    yield dbit
        else:  # terminal node (EDU)
            dbit = (''.join(prefix),
                    ''.join(suffix))
            yield (rtree, dbit)
    # return a list but we could return the generator itself
    return list(dbit for dbit in _discourse_bits(rtree))


def edu_to_tokens(rst_tree, ptb_trees):
    """Compute a matching from EDUs to tokens."""

    # tokens
    tokens = itertools.chain.from_iterable(ptree.leaves()
                                           for ptree in ptb_trees)
    # EDUs
    edus = rst_tree.leaves()
    # compute mapping
    edu2toks = defaultdict(list)
    cur_tok = tokens.next()  # init
    for edu in edus:
        while edu.span.encloses(cur_tok.span):
            edu2toks[edu].append(cur_tok)
            # print '==', cur_tok
            try:
                cur_tok = tokens.next()
            except StopIteration:  # no more token
                return edu2toks
    else:  # we should not reach here: tokens should have been exhausted
        return edu2toks


def koweys_edu_to_tokens(rst_tree, doc_ptb_trees):
    """Kowey's version of edu_to_tokens.
    
    Adapted from educe.rst_dt.learning.features._ptb_stuff()
    for comparison.
    """
    edus = rst_tree.leaves()
    
    edu2toks = dict()
    for edu in edus:
        ptb_trees = [t for t in doc_ptb_trees
                     if t.text_span().overlaps(edu.text_span())]
        all_tokens = itertools.chain.from_iterable(t.leaves()
                                                   for t in ptb_trees)
        ptb_tokens = [tok for tok in all_tokens
                      if tok.text_span().overlaps(edu.text_span())]
        edu2toks[edu] = ptb_tokens
    return edu2toks


def discourse_bits_by_token(rst_tree, ptb_trees):
    """Get the discourse bit for each token of each PTB tree."""

    # get the discourse bit for each EDU
    dbits = discourse_bits(rst_tree)
    # get the EDU to tokens mapping
    edu2toks = edu_to_tokens(rst_tree, ptb_trees)
    # init result: '*' as default value
    tokens = itertools.chain.from_iterable(ptree.leaves()
                                           for ptree in ptb_trees)
    result = {token: '*' for token in tokens}
    # compute and store discourse bits
    for edu, dbit in dbits:
        toks = edu2toks[edu]
        prefixes = ['' for _ in toks]
        prefixes[0] = dbit[0]
        suffixes = ['' for _ in toks]
        suffixes[-1] = dbit[1]
        for tok_id, tok in enumerate(toks):
            result[tok] = '{p}*{s}'.format(p=prefixes[tok_id],
                                           s=suffixes[tok_id])
    # return a dict: Token -> str
    return result


def dump_conll_rst(doc_id, rst_tree, ptb_trees, f):
    """Dump a discourse annotated document to a CoNLL-RST formatted file."""
    doc = doc_id.doc
    subdoc = doc_id.subdoc if doc_id.subdoc is not None else '0'
    header = ['#begin document ({doc});'.format(doc=doc),
              ' part {subdoc}'.format(subdoc=subdoc)]
    header = ''.join(header)
    f.write('{h}\n'.format(h=header))
    # content
    dbits = discourse_bits_by_token(rst_tree, ptb_trees)
    # 
    for ptree_id, ptree in enumerate(ptb_trees):
        tokens = ptree.leaves()
        pbits = parse_bits(ptree)  # parse bits
        for token_id, token in enumerate(tokens):
            parse_bit = pbits[token_id]
            disco_bit = dbits[token]
            #
            s = [doc,
                 subdoc,
                 token_id,
                 token.word,
                 token.tag,
                 parse_bit,
                 '_',  # lemma
                 '_',  # pred_frameset_id
                 '_',  # word_sense
                 disco_bit]
            s = '\t'.join(str(field) for field in s)
            f.write('{s}\n'.format(s=s))
        f.write('\n')  # a blank line after each sentence
    # 
    footer = '#end document'
    f.write('{z}\n'.format(z=footer))


def load_conll_rst(f):
    """Load a discourse annotated document from a CoNLL-RST formatted file."""
    _header_re = re.compile(r'#begin document \((?P<doc>.*)\); part (?P<subdoc>.*)')
    # init accumulators
    comments = []
    ptb_trees_parts = [[]]
    rst_tree_parts = []
    # parse file
    for line in f:
        line = line.strip()  # remove leading and trailing whitespaces
        if not line:  # empty line: sentence separator
            ptb_trees_parts.append([])  # new PTB tree
            continue
        elif line.startswith('#'):  # comment line
            comments.append(line)  # store comments
            # parse header
            if line.startswith('#begin document ('):
                match = _header_re.match(line)
                if not match:
                    raise Exception()
                doc = match.group('doc')
                subdoc = match.group('subdoc')
                doc_id = educe.corpus.FileId(doc=doc,
                                             subdoc=subdoc,
                                             stage='discourse',
                                             annotator='unknown')
            continue
        else:  # normal line
            s = line.split('\t')
            # PTB tree
            ptb_tree_part = (s[3], s[4], s[5])  # (word, tag, parse_bit)
            ptb_trees_parts[-1].append(ptb_tree_part)
            # RST tree
            rst_tree_part = (s[3], s[9])  # (word, discourse_bit)
            rst_tree_parts.append(rst_tree_part)
    else:
        # rebuild PTB trees
        ptb_trees = []
        for ptree in ptb_trees_parts:
            # skip empty lists of tree parts
            if not ptree:
                continue
            # normal processing
            ptree_str = []
            for ptree_part in ptree:
                word_tag = '({tag} {word})'.format(tag=ptree_part[0],
                                                   word=ptree_part[1])
                part_str = ptree_part[2].replace('*', word_tag)
                ptree_str.append(part_str)
            ptree_str = ''.join(ptree_str)
            # print ptree_str  # DEBUG
            ptb_tree = nltk.tree.Tree.fromstring(ptree_str)  #FIXME
            ptb_trees.append(ptb_tree)

        # rebuild RST tree
        edus = []
        for word, disco_bit in rst_tree_parts:
            if disco_bit.startswith('('):  # EDU start
                disco_bit = disco_bit.replace('*', '')
                edus.append(([disco_bit], [word]))
            elif disco_bit.startswith('*'):  # EDU inside or end
                edus[-1][1].append(word)
                if disco_bit[1:]:
                    edus[-1][0].append(disco_bit[1:])
            else:
                print 'Dunno what to do with this:', disco_bit
        edus_str = ['{p} _!{t}_! {s}'.format(p=edu[0][0],
                                             t=' '.join(edu[1]),
                                             s=''.join(edu[0][1:]))
                    for edu in edus]
        # TODO: produce a proper RST DT string that can be parsed by
        # educe.rst_dt.parse.parse_rst_dt_tree()
        rst_tree_str = '\n'.join(edus_str)
        #
        if False:
            rst_tree_str = ''.join(rst_tree_str)
            rst_tree = parse_rst_dt_tree(rst_tree_str)
        else:
            # print rst_tree_str  # DEBUG
            rst_tree = None
    # return a bundle
    return (doc_id, rst_tree, ptb_trees)


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
    
