"""This submodule implements Discourse Segmented Lexicalized Syntactic Trees.

A Discourse Segmented Lexicalized Syntactic Tree (DS-LST) is a lexicalized
syntactic parse tree augmented with EDU boundaries.

References
----------
Soricut and Marcu, 2003, Sentence Level Discourse Parsing using Syntactic
and Lexical Information, Proceedings of HLT-NAACL 2003.
"""

from collections import deque

from nltk.tree import Tree


# helper
def find_edu_head(tree, hwords, wanted):
    """Find the highest node in tree whose head word is in wanted.

    Returns
    -------
    tpos_hn: tuple of integers
        the tree position of the highest node whose head word is in
        ``wanted``, or None if no such node could be found
    """
    # prune wanted to prevent punctuation from becoming the head of an EDU
    nohead_tags = set(['.', ',', "''", "``"])
    wanted = set([tp for tp in wanted
                  if tree[tp].tag not in nohead_tags])

    # find the highest occurrence of any of the words in wanted
    # top-down traversal of the tree,
    # use a queue initialized with the tree pos of the root node: ()
    all_treepos = deque([()])
    while all_treepos:
        cur_treepos = all_treepos.popleft()
        cur_subtree = tree[cur_treepos]
        # if the head word is in wanted, return the current position as
        # the head node, otherwise add the daughter nodes to the queue
        if hwords[cur_treepos] in wanted:
            return cur_treepos
        elif isinstance(cur_subtree, Tree):
            c_treeposs = [tuple(list(cur_treepos) + [c_idx])
                          for c_idx, c in enumerate(cur_subtree)]
            all_treepos.extend(c_treeposs)
        else:  # don't try to recurse if the current subtree is a Token
            pass
    return None


# WIP
class DSLST(object):
    """Representation of a DS-LST"""

    def __init__(self, lstree, edus):
        self.lstree = lstree
        self.edus = edus
        self._align()

    def _align(self):
        """Internal helper to align the parse tree with the EDUs"""
        edus = self.edus
        ptree = self.lstree.ed_tree
        pheads = self.lstree.tpos_head

        treepos_words = []
        treepos_head = []
        for edu in edus:
            # align EDU boundaries with tree leaves, i.e. words
            # TODO I can probably re-use code from document_plus.tkd_tokens
            tpos_words = [tpos for tpos in ptree.treepositions('leaves')
                          if ptree[tpos].overlaps(edu)]
            treepos_words.append(tpos_words)
            # store the tree position of the head node
            tpos_hd = find_edu_head(ptree, pheads, tpos_words)
            treepos_head.append(tpos_hd)

        self.treepos_words = treepos_words
        self.treepos_head = treepos_head

    def dominance_set(self):
        """Get the dominance set of this DS-LST

        Returns
        -------
        dom_set: TODO type
            TODO description
        """
        pass  # TODO

    def exception_edu(self):
        """Get the exception EDU: head node is the root of the syn tree"""
        pass  # TODO: return type: EDU or (rel? abs?) EDU idx?

    def head_word(self, edu):
        """Word with the highest occurrence as a lexical head, from edu"""
        pass  # TODO: return type: treepos or word?

    def _treepos_head_node(self, rel_edu_idx):
        """Get the tree position of the head node for this EDU"""
        # RESUME HERE

    def head_node_label(self, rel_edu_idx):
        """Label of the head node of the EDU"""
        tpos_hn = self.treepos_head[rel_edu_idx]
        hn_lbl = self.lstree.ed_tree[tpos_hn].label()
        return hn_lbl

    def head_word(self, rel_edu_idx):
        """Head word of the EDU"""
        tpos_hn = self.treepos_head[rel_edu_idx]
        tpos_hw = self.lstree.tpos_head

    def _find_nh(self, edu):
        """helper"""
        pass  # TODO
