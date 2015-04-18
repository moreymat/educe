"""Experimental features.

"""

from __future__ import print_function

import re
import itertools

from educe.external.postag import Token
from educe.internalutil import treenode
from educe.learning.keys import Substance
from .base import DocumentPlusPreprocessor

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

SINGLE_SYNTAX = [
    ('SYN_hlabel', Substance.DISCRETE),
    ('SYN_hword', Substance.DISCRETE),
#    ('SYN', Substance.BASKET),
]


def extract_single_syntax(edu_info):
    """syntactic features for the EDU"""
    try:
        ds_lst = edu_info['ds_lst']
    except KeyError:
        return

    # TODO: rewrite, call functions on ds_lst
    if tpos_hn is not None:
        hlabel = ptree[tpos_hn].label()
        hword = ptree[pheads[tpos_hn]].word

        if False:
            # DEBUG
            edu = edu_info['edu']
            print('edu: ', edu.text())
            print('hlabel: ', hlabel)
            print('hword: ', hword)
            print('======')

        yield ('SYN_hlabel', hlabel)
        yield ('SYN_hword', hword)
    # end TODO


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

    # absolute distance
    abs_dist = abs(edu_idx1 - edu_idx2)
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
    offset1 = edu_info1['edu_idx_in_sent']
    offset2 = edu_info2['edu_idx_in_sent']
    if offset1 is not None and offset2 is not None:
        # offset diff
        yield ('offset_diff', offset1 - offset2)
        yield ('offset_diff_div3', (offset1 - offset2) / 3)
        # offset pair
        yield ('offset_pair', (offset1, offset2))
        yield ('offset_div3_pair', (offset1 / 3, offset2 / 3))

    # rev_offset features
    rev_offset1 = edu_info1['edu_rev_idx_in_sent']
    rev_offset2 = edu_info2['edu_rev_idx_in_sent']
    if rev_offset1 is not None and rev_offset2 is not None:
        yield ('rev_offset_diff', rev_offset1 - rev_offset2)
        yield ('rev_offset_diff_div3', (rev_offset1 - rev_offset2) / 3)
        yield ('rev_offset_pair', (rev_offset1, rev_offset2))
        yield ('rev_offset_div3_pair', (rev_offset1 / 3, rev_offset2 / 3))

    # sentenceID
    if sent_id1 is not None and sent_id2 is not None:
        yield ('same_sentence', (sent_id1 == sent_id2))

        # current best config: rel_dist + L/R_bools
        # abs_dist does not seem to work well for inter-sent

        # rel dist
        yield ('dist_sent', sent_id1 - sent_id2)

        # L/R booleans
        if sent_id1 < sent_id2:  # right attachment (gov < dep)
            yield ('sent_right', True)
        elif sent_id1 > sent_id2:  # left attachment
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
        ds_lst1 = edu_info1['ds_lst']
        ds_lst2 = edu_info2['ds_lst']
    except KeyError:
        return

    # generate DS-LST features for intra-sentential
    if ds_lst1 == ds_lst2:
        ds_lst = ds_lst1

        # TODO: rewrite, call functions on ds_lst

        # EDU1: get head node and its head word, plus attachment node and
        # its head word
        if tpos_hn1 is not None:
            hlabel1 = ptree[tpos_hn1].label()
            hword1 = ptree[pheads[tpos_hn1]].word
            # if the head node is not the root of the syn tree,
            # there is an attachment node
            if tpos_hn1 != ():
                tpos_an1 = tpos_hn1[:-1]
                alabel1 = ptree[tpos_an1].label()
                tpos_aw1 = pheads[tpos_an1]
                aword1 = ptree[tpos_aw1].word

        # EDU2: ibid
        if tpos_hn2 is not None:
            hlabel2 = ptree[tpos_hn2].label()
            hword2 = ptree[pheads[tpos_hn2]].word
            # if the head node is not the root of the syn tree,
            # there is an attachment node
            if tpos_hn2 != ():
                tpos_an2 = tpos_hn2[:-1]
                alabel2 = ptree[tpos_an2].label()
                tpos_aw2 = pheads[tpos_an2]
                aword2 = ptree[tpos_aw2].word

        # EXPERIMENTAL
        #
        # EDU 2 > EDU 1
        if (tpos_hn1 != () and
            tpos_aw1 in tpos_words2):
            # dominance relationship: 2 > 1
            yield ('SYN_dom_2', True)
            # attachment label and word
            yield ('SYN_alabel', alabel1)
            yield ('SYN_aword', aword1)
            # head label and word
            yield ('SYN_hlabel', hlabel1)
            yield ('SYN_hword', hword1)

        # EDU 1 > EDU 2
        if (tpos_hn2 != () and
            tpos_aw2 in tpos_words1):
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
