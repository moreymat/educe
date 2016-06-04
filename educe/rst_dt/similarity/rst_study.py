# -*- coding: utf-8 -*-
"""A study of some semantic similarities on the RST corpus.

For each EDU pair on a text, check similarity wrt rhetorical relation
(or None).

This code adapts and extends on
"Word Mover’s Distance in Python" by vene & Matt Kusner [1]_.

References
----------
.. [1] http://vene.ro/blog/word-movers-distance-in-python.html
"""

# Authors: Philippe Muller <philippe.muller@irit.fr>
#          Mathieu Morey <mathieu.morey@irit.fr>

from __future__ import print_function

import argparse
from collections import defaultdict
import itertools
import os
import sys

import numpy as np
from joblib import Parallel, delayed

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import euclidean_distances
from sklearn.preprocessing import normalize

from pyemd import emd

from educe.metrics.wmd import load_embedding
from educe.rst_dt.annotation import SimpleRSTTree
from educe.rst_dt.corpus import Reader
from educe.rst_dt.deptree import RstDepTree


# relative to the educe docs directory
# was: DATA_DIR = '/home/muller/Ressources/'
DATA_DIR = os.path.join(
    os.path.dirname(__file__),
    '..', '..', '..',
    'data',  # alt: '..', '..', 'corpora'
)
RST_DIR = os.path.join(DATA_DIR, 'rst_discourse_treebank', 'data')
RST_CORPUS = {
    'train': os.path.join(RST_DIR, 'RSTtrees-WSJ-main-1.0', 'TRAINING'),
    'test': os.path.join(RST_DIR, 'RSTtrees-WSJ-main-1.0', 'TEST'),
    'double': os.path.join(RST_DIR, 'RSTtrees-WSJ-double-1.0'),
}


def wmd(i, j):
    """Compute the Word Mover's Distance between two EDUs.

    This presupposes the existence of two global variables:
    * `edu_vecs` is a sparse 2-dimensional ndarray where each row
    corresponds to the vector representation of an EDU,
    * `D_common` is a dense 2-dimensional ndarray that contains
    the euclidean distance between each pair of word embeddings.

    Parameters
    ----------
    i : int
        Index of the first EDU.
    j : int
        Index of the second EDU.

    Returns
    -------
    s : np.double
        Word Mover's Distance between EDUs i and j.
    """
    # EMD is extremely sensitive on the number of dimensions it has to
    # work with ; keep only the dimensions where at least one of the
    # two vectors is != 0
    union_idx = np.union1d(edu_vecs[i].indices, edu_vecs[j].indices)
    # EMD segfaults on incorrect parameters:
    # * if both vectors (and thus the distance matrix) are all zeros,
    # return 0.0 (consider they are the same)
    if not np.any(union_idx):
        return 0.0
    D_minimal = D_common[np.ix_(union_idx, union_idx)]
    bow_i = edu_vecs[i, union_idx].A.ravel()
    bow_j = edu_vecs[j, union_idx].A.ravel()
    # NB: emd() has an additional named parameter: extra_mass_penalty
    # pyemd by default sets it to -1, i.e. the max value in the distance
    # matrix
    return emd(bow_i, bow_j, D_minimal)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Study the RST corpus')
    parser.add_argument('outfile', nargs='?', type=argparse.FileType('wb'),
                        default=sys.stdout,
                        help='output file')
    parser.add_argument('--corpus_sel', default='double',
                        choices=['double', 'train', 'test'],
                        help='corpus selection')
    parser.add_argument('--pairs', default='related',
                        choices=['related', 'all', 'docs', 'paras', 'sents'],
                        help='selection of EDU pairs to examine')
    parser.add_argument('--lbl_fn', default='gov_dep',
                        choices = ['gov_dep', 'r1_r2'],
                        help='label function for pairs')
    # parameters for CountVectorizer
    # NB: the following defaults differ from the standard ones in
    # CountVectorizer.
    # As of 2016-03-16, we define our defaults to be:
    # * strip accents='unicode'
    # * lowercase=False
    # * stop_words='english'
    parser.add_argument('--strip_accents', default='unicode',
                        choices=['ascii', 'unicode', 'None'],
                        help='preprocessing: method to strip accents')
    parser.add_argument('--lowercase', action='store_true',
                        help='preprocessing: lowercase')
    parser.add_argument('--stop_words', default='english',
                        choices=['english', 'None'],  # TODO: add "list"
                        help='preprocessing: filter stop words')
    parser.add_argument('--scale', default='None',
                        choices=['0_1', 'None'],
                        help='scale distance to given range')
    parser.add_argument('--n_jobs', type=int, default=1,
                        help='max number of concurrently running jobs')
    parser.add_argument('--verbose', type=int, default=1,
                        help='verbosity level')
    # TODO add arguments for train and test corpora
    args = parser.parse_args()

    # * get parameters for the CountVectorizer and outfile
    # properly recast strip_accents if None
    strip_accents = (args.strip_accents if args.strip_accents != 'None'
                     else None)
    lowercase = args.lowercase
    stop_words = (args.stop_words if args.stop_words != 'None'
                  else None)
    outfile = args.outfile
    n_jobs = args.n_jobs
    verbose = args.verbose
    corpus_sel = args.corpus_sel
    sel_pairs = args.pairs
    lbl_fn = args.lbl_fn
    distance_range = (args.scale if args.scale != 'None'
                      else None)

    # * read the corpus
    rst_corpus_dir = RST_CORPUS[corpus_sel]
    rst_reader = Reader(rst_corpus_dir)
    rst_corpus = rst_reader.slurp(verbose=True)
    corpus_texts = [v.text() for k, v in sorted(rst_corpus.items())]

    # MOVE ~ WMD.__init__()
    # load word embeddings
    vocab_dict, W = load_embedding("embed")
    # end MOVE

    # MOVE ~ WMD.fit(corpus_texts?)
    # fit CountVectorizer to the vocabulary of the corpus
    vect = CountVectorizer(
        strip_accents=strip_accents, lowercase=lowercase,
        stop_words=stop_words
    ).fit(corpus_texts)
    # compute the vocabulary common to the embeddings and corpus, restrict
    # the word embeddings matrix and replace the vectorizer
    common = [word for word in vect.get_feature_names()
              if word in vocab_dict]
    W_common = W[[vocab_dict[w] for w in common]]
    vect = CountVectorizer(
        strip_accents=strip_accents, lowercase=lowercase,
        stop_words=stop_words,
        vocabulary=common, dtype=np.double
    ).fit(corpus_texts)
    # compute the distance matrix between each pair of word embeddings
    print('Computing the distance matrix between each pair of embeddings...',
          file=sys.stderr)
    D_common = euclidean_distances(W_common)
    D_common = D_common.astype(np.double)
    # optional: scale distances to range (0, 1)
    if distance_range is not None:
        D_common /= D_common.max()
    print('done', file=sys.stderr)
    # end MOVE fit()

    # MOVE ~ WMD.transform(params?)
    # print header to file: list parameters used for this run
    # NB: this should really be a dump of the state of the *WMD* object
    params = {
        'sel_pairs': sel_pairs,
        'lbl_fn': lbl_fn,
        'corpus': os.path.relpath(rst_corpus_dir, start=DATA_DIR),
        'strip_accents': strip_accents,
        'lowercase': lowercase,
        'stop_words': stop_words,
        'n_jobs': n_jobs,
        'verbose': verbose,
    }
    print('# parameters: ({})'.format(params),
          file=outfile)

    # do the real job
    corpus_items = [x for x in sorted(rst_corpus.items())
                    if not x[0].doc.startswith('file')]
    # WIP exclude file* files for paras
    doc_keys = [key.doc for key, doc in corpus_items]
    # WIP
    if sel_pairs == 'docs':
        # pairs of documents
        doc_txts = [doc.text() for key, doc in corpus_items]
        # NB: we need to keep name "edu_vecs" so that wmd() has the right
        # ref
        # FIXME change wmd() and the code below to favor a generic name
        edu_vecs = vect.transform(doc_txts)
        edu_vecs = normalize(edu_vecs, norm='l1', copy=False)
        # compute WMD for all pairs of docs ; WMD is symmetric hence
        # combinations()
        doc_pairs_idx = list(itertools.combinations(
            range(len(corpus_items)), r=2))
        if False:  # DEBUG
            doc_pairs_wmd = Parallel(n_jobs=n_jobs, verbose=verbose)(
                delayed(wmd)(doc1_idx, doc2_idx)
                for doc1_idx, doc2_idx
                in doc_pairs_idx
            )
        else:
            doc_pairs_wmd = (wmd(doc1_idx, doc2_idx)
                             for doc1_idx, doc2_idx
                             in doc_pairs_idx)

        wmd_strs = (
            ("%s::%s::%.5f" %
             (doc_keys[doc1_idx], doc_keys[doc2_idx], sim))
            for (doc1_idx, doc2_idx), sim
            in itertools.izip(doc_pairs_idx, doc_pairs_wmd)
        )
        print('\n'.join(wmd_strs), file=outfile)
    elif sel_pairs == 'paras':
        # pairs of paragraphs
        doc_txts = [doc.text() for key, doc in corpus_items]
        doc_paras = [(doc_key, para_idx, para)
                     for doc_key, doc_txt in zip(doc_keys, doc_txts)
                     for para_idx, para in enumerate(doc_txt.split('\n\n'))]
        para_keys = [(doc_key, para_idx)
                     for doc_key, para_idx, para in doc_paras]
        paras = [para for doc_key, para_idx, para in doc_paras]

        # TODO features to detect lists (parallel structures, enumerations)
        # WIP trigger for elaboration-set-member(List ... List) ?
        def is_unique_sentence_colon(s):
            """True if a string is a unique sentence ending with a colon.

            For raw text files, unique sentence means no '\n'.
            """
            s = s.strip()
            return s[-1] == ':' and '\n' not in s

        for doc_key, para_idx, para in doc_paras:
            # dumb tokenizer
            para_toks = [x for x in para.split()]
            # dumb sentence splitter
            para_sents = para.split('\n')
            if (any(is_unique_sentence_colon(x) for x in para_sents)
                or para_toks[0] in ['1.', '2.', '3.', '4.', '5.']
                or para_toks[0] in ['1)', '2)', '3)', '4)', '5)']
                or para_toks[0] == '--'
                or para_sents[0][-1] == '--'):
                # improve this...
                print(doc_key, para_idx)
                print(para)
                print()
        raise ValueError('Para check !')
        # end WIP sections

        # NB: we need to keep name "edu_vecs" so that wmd() has the right
        # ref
        # FIXME change wmd() and the code below to favor a generic name
        edu_vecs = vect.transform(paras)
        edu_vecs = normalize(edu_vecs, norm='l1', copy=False)
        # compute WMD for all pairs ; WMD is symmetric hence
        # combinations()
        pairs_idx = list(itertools.combinations(
            range(len(paras)), r=2))
        print('Tot. nb. jobs', len(pairs_idx))  # WIP
        pairs_wmd = Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(wmd)(elt1_idx, elt2_idx)
            for elt1_idx, elt2_idx
            in pairs_idx
        )

        wmd_strs = (
            ("%s:%s:%s:%s:%.5f" %
             (para_keys[elt1_idx][0],
              para_keys[elt1_idx][1],
              para_keys[elt2_idx][0],
              para_keys[elt2_idx][1],
              sim))
            for (elt1_idx, elt2_idx), sim
            in itertools.izip(pairs_idx, pairs_wmd)
        )
        print('\n'.join(wmd_strs), file=outfile)
    elif sel_pairs == 'sents':
        # pairs of sentences
        doc_txts = [doc.text() for key, doc in corpus_items]
        doc_sents = [(doc_key, sent_idx, sent)
                     for doc_key, doc_txt in zip(doc_keys, doc_txts)
                     for sent_idx, sent in enumerate(
                             x for x in doc_txt.split('\n')
                             if x.strip())]
        sent_keys = [(doc_key, sent_idx)
                     for doc_key, sent_idx, sent in doc_sents]
        sents = [sent for doc_key, sent_idx, sent in doc_sents]
        # NB: we need to keep name "edu_vecs" so that wmd() has the right
        # ref
        # FIXME change wmd() and the code below to favor a generic name
        edu_vecs = vect.transform(sents)
        edu_vecs = normalize(edu_vecs, norm='l1', copy=False)
        # compute WMD for all pairs ; WMD is symmetric hence
        # combinations()
        pairs_idx = list(itertools.combinations(
            range(len(sents)), r=2))
        print('Tot. nb. jobs', len(pairs_idx))  # WIP
        pairs_wmd = Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(wmd)(elt1_idx, elt2_idx)
            for elt1_idx, elt2_idx
            in pairs_idx
        )

        wmd_strs = (
            ("%s:%s:%s:%s:%.5f" %
             (sent_keys[elt1_idx][0],
              sent_keys[elt1_idx][1],
              sent_keys[elt2_idx][0],
              sent_keys[elt2_idx][1],
              sim))
            for (elt1_idx, elt2_idx), sim
            in itertools.izip(pairs_idx, pairs_wmd)
        )
        print('\n'.join(wmd_strs), file=outfile)
    else:
        # pairs of EDUs
        doc_key_dtrees = [
            (doc_key.doc,
             RstDepTree.from_simple_rst_tree(SimpleRSTTree.from_rst_tree(doc)))
            for doc_key, doc in corpus_items
        ]
        edu_txts = list(e.text().replace('\n', ' ')
                        for doc_key, dtree in doc_key_dtrees
                        for e in dtree.edus)
        # vectorize each EDU using its text
        edu_vecs = vect.transform(edu_txts)
        # normalize each row of the count matrix using the l1 norm
        # (copy=False to perform in place)
        edu_vecs = normalize(edu_vecs, norm='l1', copy=False)

        # get all pairs of EDUs of interest, here as 4-tuples
        # (doc_key, gov_idx, dep_idx, lbl):
        # 1. select pairs of EDUs to evaluate
        if sel_pairs == 'related':
            edu_pairs_docs = [
                [(gov_idx, dep_idx) for dep_idx, gov_idx
                 in enumerate(dtree.heads[1:], start=1)]
                for doc_key, dtree in doc_key_dtrees
            ]
        else:  # all pairs
            edu_pairs_docs = [
                [(e1_idx, e2_idx) for e1_idx, e2_idx
                 in itertools.permutations(range(len(dtree.edus)), r=2)]
                for doc_key, dtree in doc_key_dtrees
            ]
        # 2. attach label(s) to each pair of EDUs
        if lbl_fn == 'gov_dep':
            edu_pairs = [
                [(doc_key, gov_idx, dep_idx,
                  (dtree.labels[dep_idx] if dtree.heads[dep_idx] == gov_idx
                   else 'UNRELATED'))
                 for (gov_idx, dep_idx) in edu_pairs_doc]
                for (doc_key, dtree), edu_pairs_doc
                in zip(doc_key_dtrees, edu_pairs_docs)
            ]
        else:  # r1, r2
            edu_pairs = [
                [(doc_key, e1_idx, e2_idx,
                  ', '.join(str(x) for x in
                            [dtree.labels[e1_idx], dtree.labels[e2_idx]]))
                 for (e1_idx, e2_idx) in edu_pairs_doc]
                for (doc_key, dtree), edu_pairs_doc
                in zip(doc_key_dtrees, edu_pairs_docs)
            ]
        # TODO add the possibility to use functions that filter out pairs,
        # like equivalence classes on pairs (ex: WMD is symmetric)
        # or implications (ex: (e1, e2, r) => (e2, e1, UNRELATED))

        # transform local index of EDU in doc into global index in the list
        # of all EDUs from all docs
        doc_lens = [0] + [len(dtree.edus)
                          for doc_key, dtree in doc_key_dtrees[:-1]]
        doc_offsets = np.cumsum(doc_lens)
        edu_pairs = [[(doc_key, gov_idx, dep_idx, lbl,
                       doc_offset + gov_idx, doc_offset + dep_idx)
                      for doc_key, gov_idx, dep_idx, lbl
                      in doc_edu_pairs]
                     for doc_offset, doc_edu_pairs
                     in zip(doc_offsets, edu_pairs)]
        edu_pairs = list(itertools.chain.from_iterable(edu_pairs))
        # WIP
        # compute the WMD between the pairs of EDUs
        edu_pairs_wmd = Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(wmd)(gov_idx_abs, dep_idx_abs)
            for doc_key, gov_idx, dep_idx, lbl, gov_idx_abs, dep_idx_abs
            in edu_pairs
        )

        wmd_strs = [
            ("%s::%s::%.5f::(%s)--(%s)" %
             (doc_key, lbl, sim, edu_txts[gov_idx_abs], edu_txts[dep_idx_abs]))
            for (doc_key, gov_idx, dep_idx, lbl, gov_idx_abs, dep_idx_abs), sim
            in zip(edu_pairs, edu_pairs_wmd)
        ]
        print('\n'.join(wmd_strs), file=outfile)
        # end MOVE transform()
