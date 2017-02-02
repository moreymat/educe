"""This module provides ways to transform lists of PairKeys to sparse vectors.
"""

# pylint: disable=invalid-name
# lots of scikit-conventional names here

from collections import defaultdict
import sys

import numpy as np


class KeyGroupVectorizer(object):
    """Transforms lists of KeyGroups to sparse vectors.

    Attributes
    ----------
    vocabulary_ : dict(str, int)
        Vocabulary mapping.
    """
    def __init__(self):
        self.vocabulary_ = None  # FIXME should be set in fit()

    def _count_vocab(self, vectors, fixed_vocab=False):
        """Create sparse feature matrices and shared vocabulary.

        Parameters
        ----------
        vectors : list of list of KeyGroup
            List of feature matrices, one list per doc, one line per
            sample.

        fixed_vocab : boolean, defaults to False
            If True, use the vocabulary that hopefully has already been
            set during `fit()`.

        Returns
        -------
        vocabulary : dict(str, int)
            Mapping from features to integers.

        X : list of list of list of tuple(int, float)
            List of feature matrices.
        """
        if fixed_vocab:
            vocabulary = self.vocabulary_
        else:
            # every time a new value is encountered, add it to the vocabulary
            vocabulary = defaultdict()
            vocabulary.default_factory = vocabulary.__len__

        # accumulate features from every vec
        feature_acc = []
        # thus we need to remember where each pair of EDUs and each document
        # begins
        row_ptr = []
        row_ptr.append(0)
        doc_ptr = []
        doc_ptr.append(0)

        print('fit vocab')  # DEBUG
        for vecs in vectors:
            for vec in vecs:
                for feature, featval in vec.one_hot_values_gen():
                    try:
                        feature_idx = vocabulary[feature]
                        feature_acc.append((feature_idx, featval))
                    except KeyError:
                        # ignore unknown features if fixed vocab
                        continue
                row_ptr.append(len(feature_acc))
            doc_ptr.append(row_ptr[-1])
        print('vocab done')  # DEBUG

        if not fixed_vocab:
            vocabulary = dict(vocabulary)
            if not vocabulary:
                raise ValueError("empty vocabulary")

        # build a feature count matrix out of feature_acc and row_ptr
        X = []
        doc_nxt = 0
        for i in xrange(len(row_ptr) - 1):
            current_row, next_row = row_ptr[i], row_ptr[i + 1]
            if current_row == doc_ptr[doc_nxt]:
                # start a new doc matrix
                X.append([])
                doc_nxt += 1
                print('doc ', str(i))  # DEBUG
            x = feature_acc[current_row:next_row]
            X[-1].append(x)
        # DEBUG
        n_edus = [len(y) for y in X]
        print(len(vocabulary), sys.getsizeof(vocabulary))
        print(len(X), sum(len(y) for y in X), sys.getsizeof(X))
        print(sum(nb_edus * (nb_edus - 1) for nb_edus in n_edus))
        raise ValueError('woopti')
        # end DEBUG
        return vocabulary, X

    def fit_transform(self, vectors):
        """Learn the vocabulary dictionary and return instances
        """
        vocabulary, X = self._count_vocab(vectors, fixed_vocab=False)
        self.vocabulary_ = vocabulary
        return X

    def transform(self, vectors):
        """Transform documents to EDU pair feature matrix.

        Extract features out of documents using the vocabulary
        fitted with fit.
        """
        _, X = self._count_vocab(vectors, fixed_vocab=True)
        return X
