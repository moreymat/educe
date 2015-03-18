"""Experimental features.

"""

from __future__ import print_function

from collections import deque
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
    ('num_tokens_div5', Substance.CONTINUOUS),
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
        offset = edu_info['edu_idx_in_sent']
        if offset is not None:
            yield ('num_edus_from_sent_start', offset)

        rev_offset = edu_info['edu_rev_idx_in_sent']
        if rev_offset is not None:
            yield ('num_edus_to_sent_end', rev_offset)

        # position of sentence in doc
        sent_id = edu_info['sent_idx']
        if sent_id is not None:
            yield ('sentence_id', sent_id)
    except KeyError:
        pass

    try:
        rev_offset_para = edu_info['edu_rev_idx_in_para']
        if rev_offset_para is not None:
            yield ('num_edus_to_para_end', rev_offset_para)
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
def find_edu_head(tree, hwords, wanted):
    """Find the word with highest occurrence in the lexicalized tree

    Return a pair of treepositions (head node, head word), or None if
    no occurrence of any word in wanted was found.
    """
    # prune wanted to prevent punctuation from becoming the head of an EDU
    nohead_tags = set(['.', ',', "''", "``"])
    wanted = set([tp for tp in wanted
                  if tree[tp].tag not in nohead_tags])

    all_treepos = deque([()])  # init with root treepos: ()
    while all_treepos:
        cur_treepos = all_treepos.popleft()
        cur_tree = tree[cur_treepos]
        cur_hw = hwords[cur_treepos]
        if cur_hw in wanted:
            return (cur_treepos, cur_hw)
        elif isinstance(cur_tree, Tree):
            c_treeposs = [tuple(list(cur_treepos) + [c_idx])
                          for c_idx, c in enumerate(tree[cur_treepos])]
            all_treepos.extend(c_treeposs)
        else:  # don't try to recurse if the current subtree is a Token
            pass
    return None


SINGLE_SYNTAX = [
    ('SYN_hlabel', Substance.DISCRETE),
    ('SYN_hword', Substance.DISCRETE),
#    ('SYN', Substance.BASKET),
]


def extract_single_syntax(edu_info):
    """syntactic features for the EDU"""
    try:
        ptree = edu_info['ptree']
        pheads = edu_info['pheads']
    except KeyError:
        return

    edu = edu_info['edu']

    # tree positions (in the syn tree) of the words that are in the EDU
    tpos_leaves_edu = [tpos_leaf
                       for tpos_leaf in ptree.treepositions('leaves')
                       if ptree[tpos_leaf].overlaps(edu)]
    wanted = set(tpos_leaves_edu)
    edu_head = find_edu_head(ptree, pheads, wanted)
    if edu_head is not None:
        treepos_hn, treepos_hw = edu_head
        hlabel = ptree[treepos_hn].label()
        hword = ptree[treepos_hw].word

        if False:
            # DEBUG
            print('edu: ', edu.text())
            print('hlabel: ', hlabel)
            print('hword: ', hword)
            print('======')

        yield ('SYN_hlabel', hlabel)
        yield ('SYN_hword', hword)


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
    # feats.extend(SINGLE_SYNTAX)
    # funcs.append(extract_single_syntax)

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
    # TODO abs etc
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

    # TODO relative distance?
    # maybe not as certain classifiers need positive values

    # absolute distance
    abs_dist = abs(edu_idx1 - edu_idx2)
    yield ('dist_edus_abs', abs_dist)

    # EXPERIMENTAL boolean feature for adjacent EDUs
    if abs_dist == 1:
        yield ('adjacent_edus', True)

    # (left- and right-) oriented distances
    if edu_idx1 < edu_idx2:  # right attachment (gov before dep)
        yield ('dist_edus_right', abs_dist)
    else:
        yield ('dist_edus_left', abs_dist)


# features on document structure: paragraphs and sentences

PAIR_PARA = [
    ('dist_para_abs', Substance.CONTINUOUS),
    ('dist_para_right', Substance.CONTINUOUS),
    ('dist_para_left', Substance.CONTINUOUS),
    ('same_para', Substance.DISCRETE),
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
        abs_para_dist = abs(para_id1 - para_id2)
        yield ('dist_para_abs', abs_para_dist)

        if para_id1 < para_id2: # right attachment (gov before dep)
            yield ('dist_para_right', abs_para_dist)
        elif para_id1 > para_id2:
            yield ('dist_para_left', abs_para_dist)
        else:
            yield ('same_para', True)

        # TODO: remove and see what happens
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

    sent_id1 = edu_info1['sent_idx']
    sent_id2 = edu_info2['sent_idx']

    # offset features
    try:
        offset1 = edu_info1['edu_idx_in_sent']
        offset2 = edu_info2['edu_idx_in_sent']
    except KeyError:
        pass
    else:
        # offset diff
        yield ('offset_diff', offset1 - offset2)
        yield ('offset_diff_div3', (offset1 - offset2) / 3)
        # offset pair
        yield ('offset_pair', (offset1, offset2))
        yield ('offset_div3_pair', (offset1 / 3, offset2 / 3))

    # rev_offset features
    try:
        rev_offset1 = edu_info1['edu_rev_idx_in_sent']
        rev_offset2 = edu_info2['edu_rev_idx_in_sent']
    except KeyError:
        pass
    else:
        if rev_offset1 is not None and rev_offset2 is not None:
            yield ('rev_offset_diff', rev_offset1 - rev_offset2)
            yield ('rev_offset_diff_div3', (rev_offset1 - rev_offset2) / 3)
            yield ('rev_offset_pair', (rev_offset1, rev_offset2))
            yield ('rev_offset_div3_pair', (rev_offset1 / 3, rev_offset2 / 3))

    # sentenceID
    if sent_id1 is not None and sent_id2 is not None:
        yield ('same_sentence', (sent_id1 == sent_id2))

        yield ('dist_sent', sent_id1 - sent_id2)
        # TODO resume here
        abs_sent_dist = abs(sent_id1 - sent_id2)
        # yield ('dist_sent_abs', abs_sent_dist)
        if sent_id1 < sent_id2:  # right attachment (gov < dep)
            # yield ('dist_sent_right', abs_sent_dist)
            yield ('sent_right', True)
        else:
            # yield ('dist_sent_left', abs_sent_dist)
            yield ('sent_left', True)

        yield ('sentence_id_diff_div3', (sent_id1 - sent_id2) / 3)

    # revSentenceID
    rev_sent_id1 = edu_info1['edu_rev_idx_in_para']
    rev_sent_id2 = edu_info2['edu_rev_idx_in_para']
    if rev_sent_id1 is not None and rev_sent_id2 is not None:
        yield ('rev_sentence_id_diff', rev_sent_id1 - rev_sent_id2)
        yield ('rev_sentence_id_diff_div3',
               (rev_sent_id1 - rev_sent_id2) / 3)


# syntax

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
        ptree1 = edu_info1['ptree']
        pheads1 = edu_info1['pheads']

        ptree2 = edu_info2['ptree']
        pheads2 = edu_info2['pheads']
    except KeyError:
        return

    edu1 = edu_info1['edu']
    edu2 = edu_info2['edu']

    # generate DS-LST features for intra-sentential
    if ptree1 == ptree2:
        ptree = ptree1
        pheads = pheads1

        # find the head node of EDU1
        # tree positions (in the syn tree) of the words that are in EDU1
        tpos_leaves_edu1 = [tpos_leaf
                            for tpos_leaf in ptree.treepositions('leaves')
                            if ptree[tpos_leaf].overlaps(edu1)]
        tpos_words1 = set(tpos_leaves_edu1)
        edu1_head = find_edu_head(ptree, pheads, tpos_words1)
        if edu1_head is not None:
            treepos_hn1, treepos_hw1 = edu1_head
            hlabel1 = ptree[treepos_hn1].label()
            hword1 = ptree[treepos_hw1].word
            # if the head node is not the root of the syn tree,
            # there is an attachment node
            if treepos_hn1 != ():
                treepos_an1 = treepos_hn1[:-1]
                treepos_aw1 = pheads[treepos_an1]
                alabel1 = ptree[treepos_an1].label()
                aword1 = ptree[treepos_aw1].word

        # find the head node of EDU2
        # tree positions (in the syn tree) of the words that are in EDU2
        tpos_leaves_edu2 = [tpos_leaf
                            for tpos_leaf in ptree.treepositions('leaves')
                            if ptree[tpos_leaf].overlaps(edu2)]
        tpos_words2 = set(tpos_leaves_edu2)
        edu2_head = find_edu_head(ptree, pheads, tpos_words2)
        if edu2_head is not None:
            treepos_hn2, treepos_hw2 = edu2_head
            hlabel2 = ptree[treepos_hn2].label()
            hword2 = ptree[treepos_hw2].word
            # if the head node is not the root of the syn tree,
            # there is an attachment node
            if treepos_hn2 != ():
                treepos_an2 = treepos_hn2[:-1]
                treepos_aw2 = pheads[treepos_an2]
                alabel2 = ptree[treepos_an2].label()
                aword2 = ptree[treepos_aw2].word

        # EXPERIMENTAL
        #
        # EDU 2 > EDU 1
        if (treepos_hn1 != () and
            treepos_aw1 in tpos_words2):
            # dominance relationship: 2 > 1
            yield ('SYN_dom_2', True)
            # attachment label and word
            yield ('SYN_alabel', alabel1)
            yield ('SYN_aword', aword1)
            # head label and word
            yield ('SYN_hlabel', hlabel1)
            yield ('SYN_hword', hword1)

        # EDU 1 > EDU 2
        if (treepos_hn2 != () and
            treepos_aw2 in tpos_words1):
            # dominance relationship: 1 > 2
            yield ('SYN_dom_1', True)
            # attachment label and word
            yield ('SYN_alabel', alabel2)
            yield ('SYN_aword', aword2)
            # head label and word
            yield ('SYN_hlabel', hlabel2)
            yield ('SYN_hword', hword2)

        # TODO assert that 1 > 2 and 2 > 1 cannot happen together

        # TODO fire a feature if the head nodes of EDU1 and EDU2
        # have the same attachment node ?

    # TODO fire a feature with the pair of labels of the head nodes of EDU1
    # and EDU2 ?


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
