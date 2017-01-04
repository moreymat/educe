"""Experimental features.

"""

from __future__ import print_function

from collections import Counter, defaultdict
import itertools
import re

import numpy as np

from .base import DocumentPlusPreprocessor
from educe.ptb.annotation import strip_punctuation, syntactic_node_seq
from educe.ptb.head_finder import find_edu_head
from educe.rst_dt.lecsie import (load_lecsie_feats,
                                 LINE_FORMAT as LECSIE_LINE_FORMAT)
from educe.stac.lexicon.pdtb_markers import (load_pdtb_markers_lexicon,
                                             PDTB_MARKERS_FILE)
from educe.wordreprs.brown_clusters_acl2010 import fetch_brown_clusters


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
    word2clust = fetch_brown_clusters()[3200]  # EXPERIMENTAL
    docppp = DocumentPlusPreprocessor(token_filter=token_filter,
                                      word2clust=word2clust)
    return docppp.preprocess


# ---------------------------------------------------------------------
# single EDU features
# ---------------------------------------------------------------------

def extract_single_word(doc, du_info, para_info):
    """word features for the DU"""
    try:
        words = list(itertools.chain.from_iterable(
            edu_info['words'] for edu_info in du_info))
    except KeyError:
        return

    if not words:
        return

    yield ('ptb_word_first', words[0])
    yield ('ptb_word_last', words[-1])

    yield ('ptb_word_first2', tuple(words[:2]))
    yield ('ptb_word_last2', tuple(words[-2:]))

    # * feature combinations
    # NEW 2016-04-29
    if False:  # currently unplugged (because too many features)
        yield ('ptb_word_first_last', tuple(
            itertools.chain(words[:1], words[-1:])))
        yield ('ptb_word_first_last2', tuple(
            itertools.chain(words[:1], words[-2:])))
        yield ('ptb_word_first2_last', tuple(
            itertools.chain(words[:2], words[-1:])))


#
# typography
# WIP as of 2016-04-29
#

# list of tokens that can avoid capitalization in titles
NOCAP = set([
    'a', 'and', 'at', 'of', 'on', 'or', 'the', 'to'
])


def is_title_cased(tok_seq):
    """True if a sequence of tokens is title-cased"""
    return all(x[0].isupper() for x in tok_seq
               if x not in NOCAP and x[0].isalpha())


def is_upper_init(tok_seq):
    """True if a sequence starts with two upper-cased tokens"""
    alnum_toks = [x for x in tok_seq if x.isalnum()]
    return all(x.isupper() for x in alnum_toks[:2])


def is_upper_entire(tok_seq):
    """True if a sequence is fully upper-cased"""
    return all(x.upper() == x for x in tok_seq)


def extract_single_typo(doc, du_info, para_info):
    """typographical features for the EDU"""
    try:
        words = list(itertools.chain.from_iterable(
            edu_info['words'] for edu_info in du_info))
    except KeyError:
        return

    if not words:
        return

    yield ('is_title_cased', is_title_cased(words))
    yield ('is_upper_init', is_upper_init(words))
    yield ('is_upper_entire', is_upper_entire(words))


#
# NEW discourse markers
#
MARKER2RELS = load_pdtb_markers_lexicon(PDTB_MARKERS_FILE)


def extract_single_pdtb_markers(doc, du_info, para_info):
    """Features on the presence of PDTB discourse markers in the EDU"""
    try:
        words = list(itertools.chain.from_iterable(
            edu_info['words'] for edu_info in du_info))
    except KeyError:
        return

    if not words:
        return

    markers_inc = [marker for marker in MARKER2RELS
                   if marker.appears_in(words)]
    rels_inc = [MARKER2RELS[marker] for marker in markers_inc]
    # WIP
    # TODO accumulate occurrences of same marker?
    for marker in markers_inc:
        yield ('pdtb_marker_' + str(marker), True)
    for rel in itertools.chain.from_iterable(rels_inc):
        yield ('pdtb_marked_rel_' + rel, True)
    # TODO add info to help classifiers differentiate discourse vs
    # non-discourse use of "flexible" discourse markers, ex: "and"
    # can be clausal or NP-internal
    # * linear? (index of marker in EDU from start/end)
    # * syntactic? syn nodes above the marker(s)
# end NEW


def extract_single_pos(doc, du_info, para_info):
    """POS features for the EDU"""
    try:
        tags = list(itertools.chain.from_iterable(
            edu_info['tags'] for edu_info in du_info))
    except KeyError:
        return

    if not tags:
        return

    # * core features
    yield ('ptb_pos_tag_first', tags[0])
    yield ('ptb_pos_tag_last', tags[-1])
    # nb of occurrences of each POS tag in this EDU
    tag_cnt = Counter(tags)
    for tag, occ in tag_cnt.items():
        yield ('POS_' + tag, occ)
    # NEW feature: EDU has at least a verb
    yield ('has_vb', any(tag.startswith('VB') for tag in tags))

    # * feature combinations
    # NEW 2016-04-29
    if False:  # currently unplugged (too many features)
        yield ('ptb_pos_tag_first_last', tuple(
            itertools.chain(tags[:1], tags[-1:])))
        yield ('ptb_pos_tag_first_last2', tuple(
            itertools.chain(tags[:1], tags[-2:])))
        yield ('ptb_pos_tag_first2_last', tuple(
            itertools.chain(tags[:2], tags[-1:])))


def extract_single_brown(doc, du_info, para_info):
    """Brown cluster features for the EDU"""
    try:
        brown_clusters = list(itertools.chain.from_iterable(
            edu_info['brown_clusters'] for edu_info in du_info))
    except KeyError:
        return

    if not brown_clusters:
        return

    yield ('bc_first', brown_clusters[0])
    yield ('bc_last', brown_clusters[-1])
    # nb of occurrences of each brown cluster id in this EDU
    bc_cnt = Counter(brown_clusters)
    for bc, occ in bc_cnt.items():
        yield ('bc_' + bc, occ)


def extract_single_length(doc, du_info, para_info):
    """Sentence features for the EDU"""
    try:
        words = list(itertools.chain.from_iterable(
            edu_info['words'] for edu_info in du_info))
    except KeyError:
        return

    yield ('num_tokens', len(words))
    yield ('num_tokens_div5', len(words) / 5)


# features on document structure
def extract_single_sentence(doc, du_info, para_info):
    """Sentence features for the EDU"""
    try:
        offset = du_info[0]['edu_idx_in_sent']
        if offset is not None:
            yield ('num_edus_from_sent_start', offset)

        rev_offset = du_info[-1]['edu_rev_idx_in_sent']
        if rev_offset is not None:
            yield ('num_edus_to_sent_end', rev_offset)

        # position of sentence in doc
        sent_ids = [edu_info['sent_idx'] for edu_info in du_info]
        if not any(x is None for x in sent_ids):
            yield ('sentence_id',
                   (tuple(sorted(set(x for x in sent_ids)))
                    if len(set(sent_ids)) > 1
                    else sent_ids[0]))
        # NEW position of sentence in doc, from the end
        sent_rev_ids = [edu_info['sent_rev_idx'] for edu_info in du_info]
        if not any(x is None for x in sent_rev_ids):
            yield ('sentence_rev_id',
                   (tuple(sorted(set(x for x in sent_rev_ids)))
                    if len(set(sent_rev_ids)) > 1
                    else sent_rev_ids[0]))
    except KeyError:
        pass


def extract_single_para(doc, du_info, para_info):
    """paragraph features for the DU"""
    # position of DU in paragraph
    try:
        offset_para = du_info[0]['edu_idx_in_para']
        if offset_para is not None:
            yield ('num_edus_from_para_start', offset_para)
    except KeyError:
        pass

    try:
        rev_offset_para = du_info[-1]['edu_rev_idx_in_para']
        if rev_offset_para is not None:
            yield ('num_edus_to_para_end', rev_offset_para)
    except KeyError:
        pass

    # position of paragraph in doc
    # * from beginning
    try:
        para_idc = [edu_info['para_idx'] for edu_info in du_info]
    except KeyError:
        pass
    else:
        if not any(x is None for x in para_idc):
            yield ('paragraph_id',
                   (tuple(sorted(set(x for x in para_idc)))
                    if len(set(para_idc)) > 1
                    else para_idc[0]))
            yield ('paragraph_id_div5',
                   (tuple(sorted(set(x / 5 for x in para_idc)))
                    if len(set(para_idc)) > 1
                    else para_idc[0] / 5))
    # * from end
    try:
        para_rev_idc = [edu_info['para_rev_idx'] for edu_info in du_info]
    except KeyError:
        pass
    else:
        if not any(x is None for x in para_rev_idc):
            yield ('paragraph_rev_id',
                   (tuple(sorted(set(x for x in para_rev_idc)))
                    if len(set(para_rev_idc)) > 1
                    else para_rev_idc[0]))
    # features on the surrounding paragraph
    if False:
        # WIP as of 2016-06-10
        paras = doc.paragraphs
        para_idx = edu_info['para_idx']
        if paras is not None and para_idx is not None:
            # FIXME impute para_idx where it is missing, in particular
            # for fileX files ; this should be a function from sentence
            # idx, possibly sentence length...
            # baseline imputer: para_idx = sent_idx

            # para = paras[para_idx]
            # assert para.overlaps(edu)

            # * first and last tokens of the paragraph
            # aim: get potential triggers (trailing ':') of lists
            # and markers of list items (and topic drifts?) (leading
            # '1.', '1/', 'a-', '--' ...)
            para_toks = para_info['tokens']
            if para_toks:
                # typical case for not para_toks: tokens in RST but not
                # in PTB
                yield ('para_w0', para_toks[0].word)
                yield ('para_t0', para_toks[0].tag)
                yield ('para_w-1', para_toks[-1].word)
                yield ('para_t-1', para_toks[-1].tag)

            # * syntactic characterization of paragraph
            syn_nodes = para_info['syn_nodes']
            if syn_nodes is not None:
                yield ('SYN_para_type', tuple(x.label() for x in syn_nodes))
        # end WIP


# syntactic features


def extract_single_syntax(doc, du_info, para_info):
    """syntactic features for the DU"""
    try:
        tree_idc = [edu_info['tkd_tree_idx'] for edu_info in du_info]
    except KeyError:
        return

    if any(x is None for x in tree_idc):
        return

    # ptrees = [doc.tkd_trees[x] for x in tree_idc]
    unique_ptrees = [doc.tkd_trees[x] for x, _
                     in itertools.groupby(tree_idc)]
    # pheads = doc.lex_heads[tree_idx]

    # WIP 2016-06-02: type of sentence, hopefully informative for non-S
    yield ('SYN_sent_type',
           (tuple(ptree.label() for ptree in unique_ptrees)
            if len(unique_ptrees) > 1
            else unique_ptrees[0].label()))

    # spanning nodes for the EDU
    syn_nodes = []
    syn_nodes_nopunc = []
    for tree_idx, grp in itertools.groupby(
            enumerate(tree_idc), key=lambda x: x[1]):
        # store the relative index of each EDU in this group, for
        # retrieval of their tokens
        rel_idc = [x[0] for x in grp]
        # group EDUs from the same sentence
        ptree = doc.tkd_trees[tree_idx]
        tokens = list(itertools.chain.from_iterable(
            du_info[rel_idx]['tokens'] for rel_idx in rel_idc))
        syn_nodes.extend(syntactic_node_seq(ptree, tokens))
        # variant, stripped from leading and trailing punctuations
        tokens_strip_punc = strip_punctuation(tokens)
        syn_nodes_nopunc.extend(syntactic_node_seq(ptree, tokens_strip_punc))

    if syn_nodes:
        yield ('SYN_nodes',
               tuple(x.label() for x in syn_nodes))
    # variant, stripped from leading and trailing punctuations
    if syn_nodes_nopunc:
        yield ('SYN_nodes_nopunc',
               tuple(x.label() for x in syn_nodes_nopunc))

    # currently de-activated
    if False:
        edu = edu_info['edu']
        # find EDU head
        edu_head = edu_info['edu_head']
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
    funcs = [
        # word
        extract_single_word,
        # WIP typography
        extract_single_typo,
        # discourse markers
        extract_single_pdtb_markers,
        # pos
        extract_single_pos,
        # EXPERIMENTAL Brown clusters
        # extract_single_brown,
        # length
        extract_single_length,
        # para
        extract_single_para,
        # sent
        extract_single_sentence,
    ]

    # syntax (EXPERIMENTAL)
    funcs.append(extract_single_syntax)

    def _extract_all(doc, edu_info, para_info):
        """inner helper because I am lost at sea here"""
        # TODO do this in a cleaner manner
        for fct in funcs:
            for feat in fct(doc, edu_info, para_info):
                yield feat

    # extractor
    feat_extractor = _extract_all
    return feat_extractor


# ---------------------------------------------------------------------
# EDU pairs
# ---------------------------------------------------------------------

# EXPERIMENTAL
LECSIE_FEAT_NAMES = LECSIE_LINE_FORMAT[3:]


class LecsieFeats(object):
    """Extract Lecsie features from each pair of EDUs"""

    def __init__(self, lecsie_data_dir):
        # index by (doc_name, span1_beg, span1_end, span2_beg, span2_end)
        # we linearly order the unordered pairs, for convenience later on
        self._feats = {((entry[0], entry[1], entry[2], entry[3], entry[4])
                        if entry[1] < entry[3] else
                        (entry[0], entry[3], entry[4], entry[1], entry[2])):
                       entry[5:]
                       for entry in load_lecsie_feats(lecsie_data_dir)}

    def fit(self, edu_pairs, y=None):
        """Fit the feature extractor.

        Currently a no-op.

        Parameters
        ----------
        edu_pairs: TODO
            TODO
        y: TODO, optional
            TODO

        Returns
        -------
        self: TODO
            TODO
        """
        return self

    def transform(self, edu_pairs):
        """Extract lecsie features for pairs of EDUs.

        This is a generator.

        Parameters
        ----------
        edu_pairs: TODO
            TODO

        Returns
        -------
        res: TODO
            TODO
        """
        lecsie_feats = self._feats
        for edu_info1, edu_info2 in edu_pairs:
            # retrieve doc name
            doc1 = edu_info1['edu'].origin.doc
            doc2 = edu_info2['edu'].origin.doc
            assert doc1 == doc2
            lecsie_doc_name = doc2[:-4] if doc2.endswith('.out') else doc2
            # retrieve span for both EDUs
            sent1 = edu_info1['edu'].span
            s1_beg = sent1.char_start
            s1_end = sent1.char_end
            num1 = edu_info1['edu'].num
            sent2 = edu_info2['edu'].span
            s2_beg = sent2.char_start
            s2_end = sent2.char_end
            num2 = edu_info2['edu'].num
            # WIP adjacency
            adjacent_pair = (abs(num1 - num2) == 1)
            # WIP intra- vs inter-sentential
            sent_id1 = edu_info1['sent_idx']
            sent_id2 = edu_info2['sent_idx']
            intra_sent = (sent_id1 is not None and sent_id2 is not None and
                          sent_id1 == sent_id2)

            # lecsie features are defined on unordered pairs
            # e.g. (e1, e2) and (e2, e1) have the same lecsie features
            lecsie_key = ((lecsie_doc_name, s1_beg, s1_end, s2_beg, s2_end)
                          if s1_beg < s2_beg else
                          (lecsie_doc_name, s2_beg, s2_end, s1_beg, s1_end))
            try:
                pair_lfeats = lecsie_feats[lecsie_key]
            except KeyError:
                # silently skip
                continue

            # put in a dict
            pair_lfeats = dict(zip(LECSIE_FEAT_NAMES, pair_lfeats))

            # WIP
            # composite family scores
            for family in ['wcomb', 'specificity', 'normpmi']:
                fam_scores = [fv for fn, fv in pair_lfeats.items()
                              if fn.endswith(family)]

                pair_lfeats['max_' + family] = np.nanmax(fam_scores)
                pair_lfeats['min_' + family] = np.nanmin(fam_scores)
                pair_lfeats['mean_' + family] = np.nanmean(fam_scores)
            # overall max score
            pair_lfeats['max_lecsie'] = np.nanmax(pair_lfeats.values())
            # end WIP

            # yield finite features (abstain on inf and nan)
            for fn, fv in pair_lfeats.items():
                # WIP
                # split these features: adjacent vs non-adjacent
                if adjacent_pair:
                    fn = 'adj_' + fn
                # split: intra- vs inter-sentential
                fn = fn + ('_intra' if intra_sent else '_inter')
                # end WIP

                if np.isfinite(fv):
                    yield (fn, fv)
# end EXPERIMENTAL


def extract_pair_doc(doc, du_info1, du_info2, edu_info_bwn):
    """Document-level tuple features"""
    edus1_num = [edu_info['edu'].num for edu_info in du_info1]
    edus2_num = [edu_info['edu'].num for edu_info in du_info2]

    # direction of attachment: attach_right
    attach_right = edus1_num[-1] < edus2_num[0]
    yield ('attach_right', attach_right)

    # absolute distance
    # this feature is more efficient when split in 4 features, for
    # every combination of the direction of attachment and
    # intra/inter-sentential status
    dist_edus = (edus2_num[0] - edus1_num[-1]
                 if edus1_num[-1] < edus2_num[0]
                 else edus1_num[0] - edus2_num[-1])
    yield ('dist_edus', dist_edus)


# features on document structure: paragraphs and sentences

def extract_pair_para(doc, du_info1, du_info2, edu_info_bwn):
    """Paragraph tuple features"""
    try:
        para_ids1 = [edu_info['para_idx'] for edu_info in du_info1]
        para_ids2 = [edu_info['para_idx'] for edu_info in du_info2]
    except KeyError:
        return

    if not any(x is None for x in itertools.chain(para_ids1, para_ids2)):
        if para_ids1[-1] < para_ids2[0]:  # right attachment to another para
            para_dist = para_ids1[-1] - para_ids2[0]
            yield ('dist_para_right', abs(para_dist))
        elif para_ids2[-1] < para_ids1[0]:
            para_dist = para_ids1[0] - para_ids2[-1]
            yield ('dist_para_left', abs(para_dist))
        else:
            # if the two DUs have a paragraph in common, consider they are
            # in the same
            para_dist = 0
            yield ('same_para', True)

        yield ('dist_para_abs', abs(para_dist))

        # TODO: remove and see what happens
        yield ('num_paragraphs_between_div3', abs(para_dist) / 3)


def extract_pair_sent(doc, du_info1, du_info2, edu_info_bwn):
    """Sentence tuple features"""

    sent_ids1 = [edu_info['sent_idx'] for edu_info in du_info1]
    sent_ids2 = [edu_info['sent_idx'] for edu_info in du_info2]

    # sentenceID
    if not any(x is None for x in itertools.chain(sent_ids1, sent_ids2)):
        # if the two DUs have at least one sentence in common, consider
        # they belong to the same sentence
        yield ('same_sentence',
               len(set(sent_ids1).intersection(set(sent_ids2))) > 0)

        # current best config: rel_dist + L/R_bools
        # abs_dist does not seem to work well for inter-sent
        if sent_ids1[-1] < sent_ids2[0]:
            # right attachment (gov < dep)
            dist_sent = sent_ids1[-1] - sent_ids2[0]
            yield ('sent_right', True)
        elif sent_ids2[-1] < sent_ids1[0]:
            # left attachment
            dist_sent = sent_ids1[0] - sent_ids2[-1]
            yield ('sent_left', True)
        else:
            dist_sent = 0

        # rel dist
        yield ('dist_sent', dist_sent)

        yield ('sentence_id_diff_div3', dist_sent / 3)

    # offset features
    # we take the offset of the first EDU of each DU (approximation?)
    offset1 = du_info1[0]['edu_idx_in_sent']
    offset2 = du_info2[0]['edu_idx_in_sent']
    if offset1 is not None and offset2 is not None:
        # offset diff
        offset_diff = offset1 - offset2
        yield ('offset_diff', offset_diff)
        yield ('offset_diff_div3', offset_diff / 3)
        # offset pair
        yield ('offset_div3_pair', (offset1 / 3, offset2 / 3))

    # rev_offset features
    # we take the rev offset of the last EDU of each DU (approximation)
    rev_offset1 = du_info1[-1]['edu_rev_idx_in_sent']
    rev_offset2 = du_info2[-1]['edu_rev_idx_in_sent']
    if rev_offset1 is not None and rev_offset2 is not None:
        rev_offset_diff = rev_offset1 - rev_offset2
        yield ('rev_offset_diff', rev_offset_diff)
        yield ('rev_offset_diff_div3', rev_offset_diff / 3)
        yield ('rev_offset_div3_pair', (rev_offset1 / 3, rev_offset2 / 3))

    # revSentenceID
    rev_sent_id1 = du_info1[-1]['edu_rev_idx_in_para']
    rev_sent_id2 = du_info2[-1]['edu_rev_idx_in_para']
    if rev_sent_id1 is not None and rev_sent_id2 is not None:
        yield ('rev_sentence_id_diff', rev_sent_id1 - rev_sent_id2)
        yield ('rev_sentence_id_diff_div3',
               (rev_sent_id1 - rev_sent_id2) / 3)


# syntax

def extract_pair_syntax(doc, du_info1, du_info2, edu_info_bwn):
    """syntactic features for the pair of EDUs"""
    try:
        tree_idc1 = [edu_info['tkd_tree_idx'] for edu_info in du_info1]
        tree_idc2 = [edu_info['tkd_tree_idx'] for edu_info in du_info2]
    except KeyError:
        return

    if any(x is None for x in itertools.chain(tree_idc1, tree_idc2)):
        return

    edus1_num = [edu_info['edu'].num for edu_info in du_info1]
    edus2_num = [edu_info['edu'].num for edu_info in du_info2]

    # determine the linear order of {EDU_1, EDU_2}
    if edus1_num[-1] < edus2_num[0]:
        du_info_l = du_info1
        du_info_r = du_info2
        tree_idc_l = tree_idc1
        tree_idc_r = tree_idc2
    else:
        du_info_l = du_info2
        du_info_r = du_info1
        tree_idc_l = tree_idc2
        tree_idc_r = tree_idc1

    # inside each DU, group EDUs from the same sentence
    # dict from int to int (index of tree, index of EDU in the DU)
    tree2edu1 = defaultdict(list)
    tree2edu2 = defaultdict(list)
    tree2edu1.update({tree_idx: [x[0] for x in grp]
                      for tree_idx, grp in itertools.groupby(
                              enumerate(tree_idc1), key=lambda x: x[1])})
    tree2edu2.update({tree_idx: [x[0] for x in grp]
                      for tree_idx, grp in itertools.groupby(
                              enumerate(tree_idc2), key=lambda x: x[1])})

    # (part of) DU1 and (part of) DU2 belong to the same sentence:
    # yield intra-sentential features
    common_tree_idc = set(tree_idc1).intersection(set(tree_idc2))
    if len(common_tree_idc) > 1:
        raise ValueError("More than one sentence common to DU1 and DU2")

    if len(common_tree_idc) == 1:
        # intra-sentential
        for tree_idx in common_tree_idc:
            ptree = doc.tkd_trees[tree_idx]
            pheads = doc.lex_heads[tree_idx]

            # relative index of each EDU in this sentence, from each DU
            edus_in_du1 = tree2edu1[tree_idx]
            edus_in_du2 = tree2edu2[tree_idx]
            # l/r
            if du_info_l == du_info1:
                edus_in_du_l = edus_in_du1
                edus_in_du_r = edus_in_du2
            else:
                edus_in_du_l = edus_in_du2
                edus_in_du_r = edus_in_du1

            # * DS-LST features
            # find the head node of DU1
            if len(edus_in_du1) == 1:
                x0 = edus_in_du1[0]
                tpos_words1 = du_info1[x0]['tpos_words']
                edu1_head = du_info1[x0]['edu_head']
            elif len(edus_in_du1) > 1:
                tpos_words1 = set(itertools.chain.from_iterable(
                    du_info1[x]['tpos_words'] for x in edus_in_du1))
                edu1_head = find_edu_head(ptree, pheads, tpos_words1)
            else:
                raise ValueError("Hu? No EDU from DU1 in the common sentence")

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
            if len(edus_in_du2) == 1:
                x0 = edus_in_du2[0]
                tpos_words2 = du_info2[x0]['tpos_words']
                edu2_head = du_info2[x0]['edu_head']
            elif len(edus_in_du2) > 1:
                tpos_words2 = set(itertools.chain.from_iterable(
                    du_info2[x]['tpos_words'] for x in edus_in_du2))
                edu2_head = find_edu_head(ptree, pheads, tpos_words2)
            else:
                raise ValueError("Hu? No EDU from DU2 in the common sentence?")

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
            if ((treepos_hn1 != () and
                 treepos_aw1 in tpos_words2)):
                # dominance relationship: 2 > 1
                yield ('SYN_dom_2', True)
                # attachment label and word
                yield ('SYN_alabel', alabel1)
                yield ('SYN_aword', aword1)
                # head label and word
                yield ('SYN_hlabel', hlabel1)
                yield ('SYN_hword', hword1)

            # EDU 1 > EDU 2
            if ((treepos_hn2 != () and
                 treepos_aw2 in tpos_words1)):
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

            # * syntactic nodes (WIP as of 2016-05-25)
            #   - interval between edu1 and edu2
            if edu_info_bwn:
                # bwn_edus = [x['edu'] for x in edu_info_bwn]
                bwn_tokens = list(itertools.chain.from_iterable(
                    x['tokens'] for x in edu_info_bwn))
                # 1. EDUs_bwn
                # spanning nodes for the interval
                syn_nodes = syntactic_node_seq(ptree, bwn_tokens)
                if syn_nodes:
                    yield ('SYN_nodes_bwn',
                           tuple(x.label() for x in syn_nodes))
                # variant: strip leading and trailing punctuations
                bwn_tokens_strip_punc = strip_punctuation(bwn_tokens)
                syn_nodes_strip = syntactic_node_seq(
                    ptree, bwn_tokens_strip_punc)
                if syn_nodes_strip:
                    yield ('SYN_nodes_bwn_nopunc',
                           tuple(x.label() for x in syn_nodes_strip))

                # 2. EDU_L + EDUs_bwn + EDU_R
                # lbwnr_edus = [edu_l] + bwn_edus + [edu_r]
                lbwnr_tokens = (
                    list(itertools.chain.from_iterable(
                        du_info_l[x]['tokens'] for x in edus_in_du_l))
                    + bwn_tokens
                    + list(itertools.chain.from_iterable(
                        du_info_r[x]['tokens'] for x in edus_in_du_r)))
                # spanning nodes
                syn_nodes = syntactic_node_seq(ptree, lbwnr_tokens)
                if syn_nodes:
                    yield ('SYN_nodes_lbwnr',
                           tuple(x.label() for x in syn_nodes))
                # variant: strip leading and trailing punctuations
                lbwnr_tokens_strip_punc = strip_punctuation(lbwnr_tokens)
                syn_nodes_strip = syntactic_node_seq(
                    ptree, lbwnr_tokens_strip_punc)
                if syn_nodes_strip:
                    yield ('SYN_nodes_lbwnr_nopunc',
                           tuple(x.label() for x in syn_nodes_strip))

                # 3. EDU_L + EDUs_bwn
                # lbwn_edus = [edu_l] + bwn_edus
                lbwn_tokens = (
                    list(itertools.chain.from_iterable(
                        du_info_l[x]['tokens'] for x in edus_in_du_l))
                    + bwn_tokens)
                # spanning nodes
                syn_nodes = syntactic_node_seq(ptree, lbwn_tokens)
                if syn_nodes:
                    yield ('SYN_nodes_lbwn',
                           tuple(x.label() for x in syn_nodes))
                # variant: strip leading and trailing punctuations
                lbwn_tokens_strip_punc = strip_punctuation(lbwn_tokens)
                syn_nodes_strip = syntactic_node_seq(
                    ptree, lbwn_tokens_strip_punc)
                if syn_nodes_strip:
                    yield ('SYN_nodes_lbwn_nopunc',
                           tuple(x.label() for x in syn_nodes_strip))

                # 4. EDUs_bwn + EDU_R
                # bwnr_edus = bwn_edus + [edu_r]
                bwnr_tokens = (
                    bwn_tokens
                    + list(itertools.chain.from_iterable(
                        du_info_r[x]['tokens'] for x in edus_in_du_r)))
                # spanning nodes
                syn_nodes = syntactic_node_seq(ptree, bwnr_tokens)
                if syn_nodes:
                    yield ('SYN_nodes_bwnr',
                           tuple(x.label() for x in syn_nodes))
                # variant: strip leading and trailing punctuations
                bwnr_tokens_strip_punc = strip_punctuation(bwnr_tokens)
                syn_nodes_strip = syntactic_node_seq(
                    ptree, bwnr_tokens_strip_punc)
                if syn_nodes_strip:
                    yield ('SYN_nodes_bwnr_nopunc',
                           tuple(x.label() for x in syn_nodes_strip))

                # TODO EDU_L + EDUs_bwn[:i], EDUs_bwn[i:] + EDUs_R ?
                # where i should correspond to the split point of the (2nd
                # order variant of the) Eisner decoder

                # TODO specifically handle interval PRN that start with a comma
                # that trails the preceding EDU ?

    # TODO fire a feature with the pair of labels of the head nodes of EDU1
    # and EDU2 ?
    else:
        ptrees1 = [doc.tkd_trees[x] for x in tree_idc1]
        # pheads1 = doc.lex_heads[tree_idx1]

        ptrees2 = [doc.tkd_trees[x] for x in tree_idc2]
        # pheads2 = doc.lex_heads[tree_idx2]

        # pair of sentence types, hopefully informative esp. for non-S
        yield ('SYN_sent_type_pair', (tuple(x.label() for x in ptrees1),
                                      tuple(x.label() for x in ptrees2)))
        # sentence types in between
        try:
            ptrees_lbwnr = (tree_idc_l
                            + [x['tkd_tree_idx'] for x in edu_info_bwn]
                            + tree_idc_r)
        except KeyError:
            pass
        else:
            ptrees_lbwnr = [doc.tkd_trees[x] for x, _
                            in itertools.groupby(ptrees_lbwnr)]
            stypes_lbwnr = [x.label() for x in ptrees_lbwnr]
            yield ('SYN_sent_type_lbwnr', tuple(stypes_lbwnr))
            yield ('SYN_sent_type_bwn', tuple(stypes_lbwnr[1:-1]))


def build_pair_feature_extractor(lecsie_data_dir=None):
    """Build the feature extractor for pairs of EDUs

    TODO: properly emit features on single EDUs ;
    they are already stored in sf_cache, but under (slightly) different
    names
    """
    funcs = [
        # feature type: 3
        extract_pair_doc,
        extract_pair_para,
        extract_pair_sent,
        # feature type: 5
        extract_pair_syntax,
    ]

    # 6
    # funcs.append(extract_pair_semantics)

    # LECSIE feats
    if lecsie_data_dir is not None:
        lecsie_feats = LecsieFeats(lecsie_data_dir)
        funcs.append(lambda e1, e2: lecsie_feats.transform([(e1, e2)]))

    def _extract_all(doc, edu_info1, edu_info2, edu_info_bwn):
        """inner helper because I am lost at sea here, again"""
        # TODO do this in a cleaner manner
        for fct in funcs:
            for feat in fct(doc, edu_info1, edu_info2, edu_info_bwn):
                yield feat

    # extractor
    feat_extractor = _extract_all
    return feat_extractor


def product_features(feats_g, feats_d, feats_gd):
    """Generate features by taking the product of features.

    Parameters
    ----------
    feats_g: dict(feat_name, feat_val)
        features of the gov EDU
    feats_d: dict(feat_name, feat_val)
        features of the dep EDU
    feats_gd: dict(feat_name, feat_val)
        features of the (gov, dep) edge

    Returns
    -------
    pf: dict(feat_name, feat_val)
        product features
    """
    pf = dict()

    # feature type: 2
    # ngram of POS, both EDUs
    try:
        pf['ptb_pos_tag_first_pairs'] = (feats_g['ptb_pos_tag_first'],
                                         feats_d['ptb_pos_tag_first'])
    except KeyError:
        pass

    # feature type: 1
    # ngram of words, both EDUs
    try:
        pf['ptb_word_first_pairs'] = (feats_g['ptb_word_first'],
                                      feats_d['ptb_word_first'])
    except KeyError:
        pass

    try:
        pf['ptb_word_last_pairs'] = (feats_g['ptb_word_last'],
                                     feats_d['ptb_word_last'])
    except KeyError:
        pass

    try:
        pf['ptb_word_first2_pairs'] = (feats_g['ptb_word_first2'],
                                       feats_d['ptb_word_first2'])
    except KeyError:
        pass

    try:
        pf['ptb_word_last2_pairs'] = (feats_g['ptb_word_last2'],
                                      feats_d['ptb_word_last2'])
    except KeyError:
        pass

    # feature type: 4
    # length, both EDUs
    try:
        pf['num_tokens_div5_pair'] = (feats_g['num_tokens_div5'],
                                      feats_d['num_tokens_div5'])
    except KeyError:
        pass

    # feature type: 3
    # position in sentence
    try:
        pf['offset_pair'] = (feats_g['num_edus_from_sent_start'],
                             feats_d['num_edus_from_sent_start'])
    except KeyError:
        pass

    try:
        pf['rev_offset_pair'] = (feats_g['num_edus_to_sent_end'],
                                 feats_d['num_edus_to_sent_end'])
    except KeyError:
        pass

    return pf


def combine_features(feats_g, feats_d, feats_gd):
    """Generate features by taking a (linear) combination of features.

    I suspect these do not have a great impact, if any, on results.

    Parameters
    ----------
    feats_g: dict(feat_name, feat_val)
        features of the gov EDU
    feats_d: dict(feat_name, feat_val)
        features of the dep EDU
    feats_gd: dict(feat_name, feat_val)
        features of the (gov, dep) edge

    Returns
    -------
    cf: dict(feat_name, feat_val)
        combined features
    """
    cf = dict()

    # length, both EDUs
    try:
        cf['num_tokens_diff_div5'] = (feats_g['num_tokens'] -
                                      feats_d['num_tokens']) / 5
    except KeyError:
        pass

    # position in sentence
    try:
        cf['offset_diff'] = (feats_g['num_edus_from_sent_start'] -
                             feats_d['num_edus_from_sent_start'])
    except KeyError:
        pass

    try:
        cf['rev_offset_diff'] = (feats_g['num_edus_to_sent_end'] -
                                 feats_d['num_edus_to_sent_end'])
    except KeyError:
        pass

    # not really linear combinations ... but this seems the least bad
    # place (for the time being)
    try:
        cf['offset_div3_pair'] = (feats_g['num_edus_from_sent_start'] / 3,
                                  feats_d['num_edus_from_sent_start'] / 3)
    except KeyError:
        pass

    try:
        cf['rev_offset_div3_pair'] = (feats_g['num_edus_to_sent_end'] / 3,
                                      feats_d['num_edus_to_sent_end'] / 3)
    except KeyError:
        pass

    # recombinations of combined features just produced
    try:
        cf['offset_diff_div3'] = cf['offset_diff'] / 3
    except KeyError:
        pass

    try:
        cf['rev_offset_diff_div3'] = cf['rev_offset_diff'] / 3
    except KeyError:
        pass

    return cf


def split_feature_space(feats_g, feats_d, feats_gd, keep_original=False,
                        split_criterion='dir'):
    """Split feature space on a criterion.

    Current supported criteria are:
    * 'dir': directionality of attachment,
    * 'sent': intra/inter-sentential,
    * 'dir_sent': directionality + intra/inter-sentential.

    Parameters
    ----------
    feats_g: dict(feat_name, feat_val)
        features of the gov EDU
    feats_d: dict(feat_name, feat_val)
        features of the dep EDU
    feats_gd: dict(feat_name, feat_val)
        features of the (gov, dep) edge
    keep_original: boolean, default=False
        whether to keep or replace the original features with the derived
        split features
    split_criterion: string
        feature(s) on which to split the feature space, options are
        'dir' for directionality of attachment, 'sent' for intra/inter
        sentential, 'dir_sent' for their conjunction

    Returns
    -------
    feats_g, feats_d, feats_gd: (dict(feat_name, feat_val))
        dicts of features with their copies

    Notes
    -----
    This function should probably be generalized and moved to a more
    relevant place.
    """
    suffix = ''

    # intra/inter sentential
    if split_criterion in ['sent', 'dir_sent']:
        try:
            intra_inter = ('intra' if feats_gd['same_sentence']
                           else 'inter')
        except KeyError:
            pass
        else:
            suffix += '_' + intra_inter

    # attachment dir
    if split_criterion in ['dir', 'dir_sent']:
        try:
            attach_dir = ('right' if feats_gd['attach_right']
                          else 'left')
        except KeyError:
            pass
        else:
            suffix += '_' + attach_dir

    if not suffix:
        return feats_g, feats_d, feats_gd

    # TODO find the right place and formulation for this, so as to
    # minimize redundancy
    if keep_original:
        feats_g.update((fn + suffix, fv)
                       for fn, fv in feats_g.items())
        feats_d.update((fn + suffix, fv)
                       for fn, fv in feats_d.items())
        feats_gd.update((fn + suffix, fv)
                        for fn, fv in feats_gd.items())
    else:
        feats_g = {(fn + suffix): fv
                   for fn, fv in feats_g.items()}
        feats_d = {(fn + suffix): fv
                   for fn, fv in feats_d.items()}
        feats_gd = {(fn + suffix): fv
                    for fn, fv in feats_gd.items()}

    return feats_g, feats_d, feats_gd
