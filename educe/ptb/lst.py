"""This submodule enables to retrieve Lexicalized Syntactic Trees.
"""

import itertools

from educe.external.parser import (ConstituencyTree)
from educe.ptb.annotation import (is_non_empty, prune_tree,
                                  strip_subcategory, transform_tree)
from educe.ptb.head_finder import find_lexical_heads


class LexicalizedSyntacticTree(object):
    """Syntactic tree where each node has a head word.

    The syntactic tree is an educe constituency tree, in which empty
    nodes are pruned and grammatical functions removed.

    Notes
    -----
    It is likely that the educification of trees should not be
    done here, but as external post-processing.
    Then this module would enable to retrieve lexicalized syntactic trees
    similar to the ones output by the Stanford Parser.
    Here is a sample ouput:
    https://mailman.stanford.edu/pipermail/parser-user/2011-May/001005.html
    """

    def __init__(self, ed_tree, lex_head, tpos_head=None):
        """Create an educified lexicalized syntactic tree.

        Parameters
        ----------
        ed_tree: educified syntactic tree
            syntactic tree that has been cleaned and educified
        lex_head: dict(tpos, (word, tag))
            mapping from tree positions in ed_tree to their lexical head
        tpos_head: dict(tpos, tpos), optional
            mapping from tree positions in ed_tree to the tree position of
            their lexical head; if none is provided, it shoule be inferred
            from lex_head
        """
        self.ed_tree = ed_tree
        self.lex_head = lex_head
        # TODO reverse-engineer tpos_head from lex_head if tpos_head is None
        self.tpos_head = tpos_head

    @classmethod
    def from_ptb_tree(cls, tree, tokens):
        """Create an educified lexicalized syntactic tree from a PTB tree

        Parameters
        ----------
        tree: PTB tree
            PTB tree
        tokens: tokens
            educe tokens (needed for educification of the tree)

        Returns
        -------
        lstree: LexicalizedSyntacticTree
            the educified LST
        """
        # apply standard cleaning to PTB tree
        # strip function tags, remove empty nodes
        tree_no_empty = prune_tree(tree, is_non_empty)
        tree_no_empty_no_gf = transform_tree(tree_no_empty,
                                             strip_subcategory)
        # educification: transform into an educe constituency tree
        tokens_iter = iter(tokens)
        leaves = tree_no_empty_no_gf.leaves()
        tslice = itertools.islice(tokens_iter, len(leaves))
        ed_tree = ConstituencyTree.build(tree_no_empty_no_gf,
                                         tslice)

        # find the head word of each constituent
        # constituents and their heads are designated by the Gorn address
        # ("tree position" in NLTK) of their node in the tree
        # this is a dict(tpos, tpos)
        tpos_head = find_lexical_heads(ed_tree)
        # map each constituent to its head word (and its POS tag)
        lex_head = {tpos_n: (ed_tree[tpos_h].word, ed_tree[tpos_h].tag)
                    for tpos_n, tpos_h in tpos_head.items()}

        # create LST
        lstree = cls(ed_tree, lex_head, tpos_head)
        return lstree

    def lexical_head_treepos(self, tpos):
        """Get the treepos of the lexical head of the constituent node at tpos

        Parameters
        ----------
        tpos: tree position
            tree position of the constituent node

        Returns
        -------
        tpos_hd: tree position
            tree position of the head of the constituent node
        """
        tpos_hd = self.tpos_head[tpos]
        return tpos_hd

    def lexical_head(self, tpos):
        """Get the lexical head of the constituent node at tpos

        Parameters
        ----------
        tpos: tree position
            tree position of the constituent node

        Returns
        -------
        word_tag: (word, tag)
            head word of the constituent node, with its POS tag
        """
        word_tag = self.lex_head[tpos]
        return word_tag
