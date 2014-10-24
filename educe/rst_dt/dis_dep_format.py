#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Authors: Mathieu Morey
# License: CeCILL-B (BSD3-like)

"""This module implements a loader and dumper for the dis-dep format.

The dis-dep format is a dependency-based alternative to the dis format
used to represent discourse trees in the RST-DT treebank.
"""

from educe.external.parser import SearchableTree
from ..internalutil import treenode


# deps is the list of dependent ids ordered by nested ids
_FORMAT = 'id text head label deps'


def _dump_dis_dep(dtree, f):
    # special depth-first, pre-order, iterator
    def _depth_first_iterator(dtree):
        """Iterate on the nodes of a dependency tree, depth-first, pre-order.

        The dependency tree is not an educe.external.parser.DependencyTree,
        but a nltk.tree.Tree whose nodes (including leaves) are
        of class RelDepNode, as produced by
        educe.rst_dt.deptree.relaxed_nuclearity_to_deptree().

        FIXME: this method contains hard-coded values (0, 'ROOT') for the
        root. The relevant tweaks should be removed when a fake root node
        is introduced.
        """
        # FIXME 0
        current = (dtree, 0)  # node, parent
        parent_stack = []
        while parent_stack or (current is not None):
            if current is not None:
                node, node_parent_id = current
                # build node representation
                rnode = treenode(node)
                node_repr = (rnode.edu.num,
                             rnode.edu.text_span(),
                             rnode.edu.text(),
                             node_parent_id,
                             # FIXME 'ROOT'
                             rnode.rel if rnode.rel is not None else 'ROOT',
                             [treenode(kid).edu.num for kid in node])
                yield node_repr
                if node:  # if node has children
                    parent_stack.extend((kid, rnode.edu.num)
                                        for kid in reversed(node[1:]))
                    current = (node[0], rnode.edu.num)
                else:
                    current = None
            else:
                current = parent_stack.pop()

    # 
    for node in sorted(list(_depth_first_iterator(dtree))):
        s_fields = [str(node[0]),  # id
                    # str(node[1]),  # text_span
                    str(node[2]),  # text
                    str(node[3]) if node[3] is not None else '0',  # parent_id
                    str(node[4]) if node[4] is not None else 'ROOT',  # label
                    str(node[5])  # ordered children
                ]
        s = '\t'.join(s_fields)
        print '{s}'.format(s=s)


def dump_dis_dep_file(dtree, f):
    """Dump a discourse dependency tree in the dis-dep format to a file."""
    _dump_dis_dep(dtree, f)
