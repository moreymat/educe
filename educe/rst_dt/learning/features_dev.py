"""Experimental features.

"""

from __future__ import print_function

import re
import itertools

from nltk.tree import Tree

from educe.external.postag import Token
from educe.internalutil import treenode
from educe.learning.keys import Substance
from .base import lowest_common_parent, DocumentPlusPreprocessor


# ---------------------------------------------------------------------
# preprocess EDUs
# ---------------------------------------------------------------------

# filter tags and tokens as in Li et al.'s parser
TT_PATTERN = r'.*[a-zA-Z_0-9].*'
TT_FILTER = re.compile(TT_PATTERN)


def token_filter_li2014(token):
    """Token filter defined in Li et al.'s parser.

    This filter only applies to tagged tokens.
    """
    return (TT_FILTER.match(token.word) is not None and
            TT_FILTER.match(token.tag) is not None)


def build_doc_preprocessor():
    """Build the preprocessor for feature extraction in each EDU of doc"""
    # TODO re-do in a better, more modular way
    token_filter = None  # token_filter_li2014
    docppp = DocumentPlusPreprocessor(token_filter)
    return docppp.preprocess


# ---------------------------------------------------------------------
# single EDU features
# ---------------------------------------------------------------------

SINGLE_WORD = [
    ('ptb_word_first', Substance.DISCRETE),
    ('ptb_word_last', Substance.DISCRETE),
    ('ptb_word_first2', Substance.DISCRETE),
    ('ptb_word_last2', Substance.DISCRETE)
]


def extract_single_word(edu_info):
    """word features for the EDU"""
    try:
        words = edu_info['words']
    except KeyError:
        return

    if words:
        yield ('ptb_word_first', words[0])
        yield ('ptb_word_last', words[-1])

    if len(words) > 1:
        yield ('ptb_word_first2', (words[0], words[1]))
        yield ('ptb_word_last2', (words[-2], words[-1]))


SINGLE_POS = [
    ('ptb_pos_tag_first', Substance.DISCRETE),
    ('ptb_pos_tag_last', Substance.DISCRETE),
    ('POS', Substance.BASKET)
]


def extract_single_pos(edu_info):
    """POS features for the EDU"""
    try:
        tags = edu_info['tags']
    except KeyError:
        return

    if tags:
        yield ('ptb_pos_tag_first', tags[0])
        yield ('ptb_pos_tag_last', tags[-1])
        for tag in tags:
            yield ('POS', tag)


SINGLE_LENGTH = [
    ('num_tokens', Substance.CONTINUOUS),
    ('num_tokens_div5', Substance.CONTINUOUS)
]


def extract_single_length(edu_info):
    """Sentence features for the EDU"""
    try:
        words = edu_info['words']
    except KeyError:
        return

    yield ('num_tokens', len(words))
    yield ('num_tokens_div5', len(words) / 5)


# features on document structure

SINGLE_SENTENCE = [
    # offset
    ('num_edus_from_sent_start', Substance.CONTINUOUS),
    # revOffset
    ('num_edus_to_sent_end', Substance.CONTINUOUS),
    # sentenceID
    ('sentence_id', Substance.CONTINUOUS),
    # revSentenceID
    ('num_edus_to_para_end', Substance.CONTINUOUS)
]


def extract_single_sentence(edu_info):
    """Sentence features for the EDU"""
    try:
        yield ('num_edus_from_sent_start', edu_info['edu_idx_in_sent'])
        yield ('num_edus_to_sent_end', edu_info['edu_rev_idx_in_sent'])
        # position of sentence in doc
        sent_id = edu_info['sent_idx']
        if sent_id is not None:
            yield ('sentence_id', sent_id)
    except KeyError:
        pass

    try:
        yield ('num_edus_to_para_end', edu_info['edu_rev_idx_in_para'])
    except KeyError:
        pass


SINGLE_PARA = [
    ('paragraph_id', Substance.CONTINUOUS),
    ('paragraph_id_div5', Substance.CONTINUOUS)
]


def extract_single_para(edu_info):
    """paragraph features for the EDU"""
    try:
        para_idx = edu_info['para_idx']
    except KeyError:
        pass
    else:
        if para_idx is not None:
            yield ('paragraph_id', para_idx)
            yield ('paragraph_id_div5', para_idx / 5)


# syntactic features

# helpers
def syn_subtree_spanning_edu(edu_info):
    """Syntactic subtree that spans the EDU

    Returns (tree, treepos) where treepos is the position of the subtree
    """
    try:
        ptrees = edu_info['ptrees']
    except KeyError:
        return None

    edu = edu_info['edu']
    # raise warning if this EDU belongs to > 1 PTB tree
    if len(ptrees) > 1:
        print('W: more than 1 tree')
        return None
    elif not ptrees:
        print('W: no tree')
        return None

    # now we can safely assume there is one PTB tree
    ptree = ptrees[0]
    # get the indices of the leaves of the tree that are in this EDU
    leaf_ids = [idx for idx, leaf in enumerate(ptree.leaves())
                if leaf.overlaps(edu)]
    # use indices to get the tree position of their lowest common parent
    idx_start = leaf_ids[0]
    idx_end = leaf_ids[-1] + 1
    assert leaf_ids == range(leaf_ids[0], leaf_ids[-1]+1)  # sanity check
    treepos_lcp = ptree.treeposition_spanning_leaves(idx_start, idx_end)

    return (ptree, treepos_lcp)


def syn_label_spanning_edu(edu_info):
    """Get the syntactic label of the lowest node spanning the EDU"""
    try:
        ptree, treepos_lcp = syn_subtree_spanning_edu(edu_info)
    except TypeError:
        # if call returns None
        return None

    spanning_subtree = ptree[treepos_lcp]
    if spanning_subtree is None:
        return None

    if isinstance(spanning_subtree, Tree):
        spanning_lbl = treenode(spanning_subtree)
    elif isinstance(spanning_subtree, Token):
        spanning_lbl = spanning_subtree.tag
    else:
        raise ValueError('spanning_subtree is neither a Tree nor a Token')

    return spanning_lbl


# helper
# TODO rewrite
def get_syntactic_labels(edu_info):
    "Syntactic labels for this EDU"
    result = []

    try:
        ptrees = edu_info['ptrees']
    except KeyError:
        return None

    edu = edu_info['edu']

    # raise warning if this EDU belongs to > 1 PTB tree
    if len(ptrees) > 1:
        w_msg = 'W: {} belongs to more than one PTB tree'
        print(w_msg.format(edu))
        print('EDU text on span {}'.format(edu.text_span()))
        print('    {}'.format(edu.text()))
        for ptree in ptrees:
            print('PTB tree on span {}'.format(ptree.text_span()))
            print('    ', ' '.join(tok.word for tok in ptree.leaves()))
        print('')
        return []
    elif len(ptrees) == 0:
        w_msg = 'W: EDU {} does not belong to any PTB tree'
        print(w_msg.format(edu))
        return []

    # now we can safely assume there is a unique PTB tree
    ptree = ptrees[0]
    # get the indices of the leaves of the tree that are in this EDU
    leaf_ids = [idx for idx, leaf in enumerate(ptree.leaves())
                if leaf.overlaps(edu)]
    # use indices to get the tree position of their lowest common parent
    idx_start = leaf_ids[0]
    idx_end = leaf_ids[-1] + 1
    assert leaf_ids == range(leaf_ids[0], leaf_ids[-1]+1)  # sanity check
    treepos_lcp = ptree.treeposition_spanning_leaves(idx_start, idx_end)

    # EDU
    tpos_leaves_edu = ((ptree, [tpos_leaf
                                for tpos_leaf in ptree.treepositions('leaves')
                                if ptree[tpos_leaf].overlaps(edu)])
                       for ptree in ptrees)
    # for each span of syntactic leaves in this EDU
    for ptree, leaves in tpos_leaves_edu:
        tpos_parent = lowest_common_parent(leaves)
        # CHECKING / RESUME HERE
        assert tpos_parent == treepos_lcp
        # end CHECKING

        # for each leaf between leftmost and rightmost, add its ancestors
        # up to the lowest common parent
        for leaf in leaves:
            for i in reversed(range(len(leaf))):
                tpos_node = leaf[:i]
                node = ptree[tpos_node]
                node_lbl = treenode(node)
                if tpos_node == tpos_parent:
                    result.append('top_' + node_lbl)
                    break
                else:
                    result.append(node_lbl)
    return result
# end rewrite


SINGLE_SYNTAX = [
    ('SYN_label', Substance.DISCRETE),
#    ('SYN', Substance.BASKET),
]


def extract_single_syntax(edu_info):
    """syntactic features for the EDU"""
    # EXPERIMENTAL
    syn_lbl = syn_label_spanning_edu(edu_info)
    if syn_lbl is not None:
        yield ('SYN_label', syn_lbl)

    # former features
    if False:
        syn_labels = get_syntactic_labels(edu_info)
        if syn_labels is not None:
            for syn_label in syn_labels:
                yield ('SYN', syn_label)


# TODO: features on semantic similarity

def build_edu_feature_extractor():
    """Build the feature extractor for single EDUs"""
    feats = []
    funcs = []

    # word
    feats.extend(SINGLE_WORD)
    funcs.append(extract_single_word)
    # pos
    feats.extend(SINGLE_POS)
    funcs.append(extract_single_pos)
    # length
    feats.extend(SINGLE_LENGTH)
    funcs.append(extract_single_length)
    # para
    feats.extend(SINGLE_PARA)
    funcs.append(extract_single_para)
    # sent
    feats.extend(SINGLE_SENTENCE)
    funcs.append(extract_single_sentence)
    # syntax (EXPERIMENTAL)
    feats.extend(SINGLE_SYNTAX)
    funcs.append(extract_single_syntax)

    def _extract_all(edu_info):
        """inner helper because I am lost at sea here"""
        # TODO do this in a cleaner manner
        for fct in funcs:
            for feat in fct(edu_info):
                yield feat

    # header
    header = feats
    # extractor
    feat_extractor = _extract_all
    # return header and extractor
    return header, feat_extractor


# ---------------------------------------------------------------------
# EDU pairs
# ---------------------------------------------------------------------

PAIR_WORD = [
    ('ptb_word_first_pairs', Substance.DISCRETE),
    ('ptb_word_last_pairs', Substance.DISCRETE),
    ('ptb_word_first2_pairs', Substance.DISCRETE),
    ('ptb_word_last2_pairs', Substance.DISCRETE),
]


def extract_pair_word(edu_info1, edu_info2):
    """word tuple features"""
    try:
        words1 = edu_info1['words']
        words2 = edu_info2['words']
    except KeyError:
        return

    # pairs of unigrams
    if words1 and words2:
        yield ('ptb_word_first_pairs', (words1[0], words2[0]))
        yield ('ptb_word_last_pairs', (words1[-1], words2[-1]))

    # pairs of bigrams
    if len(words1) > 1 and len(words2) > 1:
        yield ('ptb_word_first2_pairs', (tuple(words1[:2]),
                                         tuple(words2[:2])))
        yield ('ptb_word_last2_pairs', (tuple(words1[-2:]),
                                        tuple(words2[-2:])))


# pos
PAIR_POS = [
    ('ptb_pos_tag_first_pairs', Substance.DISCRETE),
]


def extract_pair_pos(edu_info1, edu_info2):
    """POS tuple features"""
    try:
        tags1 = edu_info1['tags']
        tags2 = edu_info2['tags']
    except KeyError:
        return

    if tags1 and tags2:
        yield ('ptb_pos_tag_first_pairs', (tags1[0], tags2[0]))


PAIR_LENGTH = [
    ('num_tokens_div5_pair', Substance.DISCRETE),
    ('num_tokens_diff_div5', Substance.CONTINUOUS)
]


def extract_pair_length(edu_info1, edu_info2):
    """Sentence tuple features"""
    try:
        words1 = edu_info1['words']
        words2 = edu_info2['words']
    except KeyError:
        return

    num_toks1 = len(words1)
    num_toks2 = len(words2)

    yield ('num_tokens_div5_pair', (num_toks1 / 5, num_toks2 / 5))
    yield ('num_tokens_diff_div5', (num_toks1 - num_toks2) / 5)


PAIR_DOC = [
    ('dist_edus_abs', Substance.CONTINUOUS),
    ('dist_edus_left', Substance.CONTINUOUS),
    ('dist_edus_right', Substance.CONTINUOUS),
]


def extract_pair_doc(edu_info1, edu_info2):
    """Document-level tuple features"""
    edu_idx1 = edu_info1['edu'].num
    edu_idx2 = edu_info2['edu'].num
    # TODO  rel_dist (no abs), but not now as certain classifiers need val>0
    abs_dist = abs(edu_idx1 - edu_idx2)
    yield ('dist_edus_abs', abs_dist)
    if edu_idx1 < edu_idx2:  # right attachment (gov before dep)
        yield ('dist_edus_right', abs_dist)
    else:
        yield ('dist_edus_left', abs_dist)


# features on document structure: paragraphs and sentences

PAIR_PARA = [
    ('first_paragraph', Substance.DISCRETE),
    ('num_paragraphs_between', Substance.CONTINUOUS),
    ('num_paragraphs_between_div3', Substance.CONTINUOUS)
]


def extract_pair_para(edu_info1, edu_info2):
    """Paragraph tuple features"""
    try:
        para_id1 = edu_info1['para_idx']
        para_id2 = edu_info2['para_idx']
    except KeyError:
        return
    if para_id1 is not None and para_id2 is not None:
        if para_id1 < para_id2:
            first_para = 'first'
        elif para_id1 > para_id2:
            first_para = 'second'
        else:
            first_para = 'same'
        yield ('first_paragraph', first_para)

        yield ('num_paragraphs_between', para_id1 - para_id2)
        yield ('num_paragraphs_between_div3', (para_id1 - para_id2) / 3)


PAIR_SENT = [
    ('offset_diff', Substance.CONTINUOUS),
    ('rev_offset_diff', Substance.CONTINUOUS),
    ('offset_diff_div3', Substance.CONTINUOUS),
    ('rev_offset_diff_div3', Substance.CONTINUOUS),
    ('offset_pair', Substance.DISCRETE),
    ('rev_offset_pair', Substance.DISCRETE),
    ('offset_div3_pair', Substance.DISCRETE),
    ('rev_offset_div3_pair', Substance.DISCRETE),
    ('same_bad_sentence', Substance.DISCRETE),
    ('sentence_id_diff', Substance.CONTINUOUS),
    ('sentence_id_diff_div3', Substance.CONTINUOUS),
    ('rev_sentence_id_diff', Substance.CONTINUOUS),
    ('rev_sentence_id_diff_div3', Substance.CONTINUOUS)
]


def extract_pair_sent(edu_info1, edu_info2):
    """Sentence tuple features"""
    # offset features
    try:
        offset1 = edu_info1['edu_idx_in_sent']
        offset2 = edu_info2['edu_idx_in_sent']
    except KeyError:
        pass
    else:
        yield ('offset_diff', offset1 - offset2)
        yield ('offset_diff_div3', (offset1 - offset2) / 3)
        yield ('offset_pair', (offset1, offset2))
        yield ('offset_div3_pair', (offset1 / 3, offset2 / 3))
    # rev_offset features
    try:
        rev_offset1 = edu_info1['edu_rev_idx_in_sent']
        rev_offset2 = edu_info2['edu_rev_idx_in_sent']
    except KeyError:
        pass
    else:
        yield ('rev_offset_diff', rev_offset1 - rev_offset2)
        yield ('rev_offset_diff_div3', (rev_offset1 - rev_offset2) / 3)
        yield ('rev_offset_pair', (rev_offset1, rev_offset2))
        yield ('rev_offset_div3_pair', (rev_offset1 / 3, rev_offset2 / 3))

    # sentenceID
    sent_id1 = edu_info1['sent_idx']
    sent_id2 = edu_info2['sent_idx']
    if sent_id1 is not None and sent_id2 is not None:
        yield ('same_sentence',
               'same' if sent_id1 == sent_id2 else 'different')
        yield ('sentence_id_diff', sent_id1 - sent_id2)
        yield ('sentence_id_diff_div3', (sent_id1 - sent_id2) / 3)

    # revSentenceID
    rev_sent_id1 = edu_info1['edu_rev_idx_in_para']
    rev_sent_id2 = edu_info2['edu_rev_idx_in_para']
    if rev_sent_id1 is not None and rev_sent_id2 is not None:
        yield ('rev_sentence_id_diff', rev_sent_id1 - rev_sent_id2)
        yield ('rev_sentence_id_diff_div3',
               (rev_sent_id1 - rev_sent_id2) / 3)


PAIR_SYNTAX = [
    ('SYN_label_pair', Substance.DISCRETE),
    # relation between spanning nodes in the syntactic tree
    ('SYN_same_span', Substance.CONTINUOUS),
    ('SYN_sisters', Substance.CONTINUOUS),
    ('SYN_embed', Substance.CONTINUOUS),
]


def extract_pair_syntax(edu_info1, edu_info2):
    """syntactic features for the pair of EDUs"""
    try:
        ptree1, treepos_lcp1 = syn_subtree_spanning_edu(edu_info1)
        ptree2, treepos_lcp2 = syn_subtree_spanning_edu(edu_info2)
    except TypeError:
        # if either call returns None, just leave
        return

    syn_lbl1 = ptree1[treepos_lcp1]
    syn_lbl2 = ptree2[treepos_lcp2]

    # EXPERIMENTAL
    if syn_lbl1 is not None and syn_lbl2 is not None:
        yield ('SYN_label_pair', (syn_lbl1, syn_lbl2))

    # if both EDUs belong to the same sentence
    if (ptree1 == ptree2):
        if treepos_lcp1 == treepos_lcp2:
            yield ('SYN_same_span', True)
        elif treepos_lcp1[:-1] == treepos_lcp2[:-1]:
            yield ('SYN_sisters', True)
        else:
            for c1, c2 in itertools.izip_longest(treepos_lcp1, treepos_lcp2):
                if c1 is None:  # c1 is a prefix for c2
                    # means c2 is dominated by c1
                    yield ('SYN_1_over_2', True)
                    break
                elif c2 is None:  # c2 is a prefix for c1
                    # means c1 is dominated by c2
                    yield ('SYN_2_over_1', True)
                    break
                elif c1 != c2:  # c1 and c2 differ
                    # means they are in separate parts of the tree
                    break

def build_pair_feature_extractor():
    """Build the feature extractor for pairs of EDUs

    TODO: properly emit features on single EDUs ;
    they are already stored in sf_cache, but under (slightly) different
    names
    """
    feats = []
    funcs = []

    # feature type: 1
    feats.extend(PAIR_WORD)
    funcs.append(extract_pair_word)
    # 2
    feats.extend(PAIR_POS)
    funcs.append(extract_pair_pos)
    # 3
    feats.extend(PAIR_DOC)
    funcs.append(extract_pair_doc)
    feats.extend(PAIR_PARA)
    funcs.append(extract_pair_para)
    feats.extend(PAIR_SENT)
    funcs.append(extract_pair_sent)
    # 4
    feats.extend(PAIR_LENGTH)
    funcs.append(extract_pair_length)
    # 5
    feats.extend(PAIR_SYNTAX)
    funcs.append(extract_pair_syntax)
    # 6
    # feats.extend(PAIR_SEMANTICS)  # NotImplemented
    # funcs.append(extract_pair_semantics)

    def _extract_all(edu_info1, edu_info2):
        """inner helper because I am lost at sea here, again"""
        # TODO do this in a cleaner manner
        for fct in funcs:
            for feat in fct(edu_info1, edu_info2):
                yield feat

    # header
    header = feats
    # extractor
    feat_extractor = _extract_all
    # return header and extractor
    return header, feat_extractor
