"""This submodule implements document vectorizers"""

import itertools
import numbers

from collections import defaultdict, Counter

from educe.rst_dt.document_plus import DocumentPlus


class DocumentLabelExtractor(object):
    """Label extractor for the RST-DT treebank."""

    def __init__(self, instance_generator,
                 unknown_label='__UNK__',
                 labelset=None):
        """
        instance_generator to enumerate the instances from a doc
        """
        self.instance_generator = instance_generator
        self.unknown_label = unknown_label
        self.labelset = labelset

    def _extract_labels(self, doc):
        """Extract a label for each EDU pair extracted from doc

        Returns ([edu_pairs], [labels])
        """
        edu_pairs = self.instance_generator(doc)
        # extract one label per EDU pair
        labels = doc.relations(edu_pairs)
        return labels

    def _instance_labels(self, raw_documents):
        """Extract label of instances, restricted to labelset"""
        labelset = self.labelset_
        # unknown labels
        unk_lab = self.unknown_label
        unk_lab_id = labelset[unk_lab]

        analyze = self.build_analyzer()
        for doc in raw_documents:
            doc_labels = analyze(doc)
            for lab in doc_labels:
                try:
                    lab_id = labelset[lab]
                    yield lab_id
                except KeyError:
                    yield unk_lab_id

    def _learn_labelset(self, raw_documents, fixed_labelset):
        """Learn the labelset"""
        unk_lab = self.unknown_label

        if fixed_labelset:
            labelset = self.labelset_
            unk_lab_id = labelset[unk_lab]
        else:
            # add a new value when a new label is seen
            labelset = defaultdict()
            labelset.default_factory = labelset.__len__
            # the id of the unknown label should be 0
            unk_lab_id = labelset[unk_lab]

        analyze = self.build_analyzer()
        for doc in raw_documents:
            doc_labels = analyze(doc)
            for lab in doc_labels:
                try:
                    lab_id = labelset[lab]
                except KeyError:
                    continue

        if not fixed_labelset:
            # disable defaultdict behaviour
            labelset = dict(labelset)
            if not labelset:
                raise ValueError('empty labelset')

        return labelset

    def decode(self, doc):
        """Decode the input into a DocumentPlus

        doc is an educe.corpus.FileId
        """
        if not isinstance(doc, DocumentPlus):
            # doc = self.decoder(doc)
            raise ValueError('doc should be a DocumentPlus')
        return doc

    def build_analyzer(self):
        """Return a callable that extracts feature vectors from a doc"""
        return lambda doc: self._extract_labels(self.decode(doc))

    def _validate_labelset(self):
        """Validate labelset"""
        labelset = self.labelset
        if labelset is not None:
            if not labelset:
                raise ValueError('empty labelset passed to fit')
            self.fixed_labelset_ = True
            self.labelset_ = dict(labelset)
        else:
            self.fixed_labelset_ = False

    def fit(self, raw_documents):
        """Learn a labelset from the documents"""
        self._validate_labelset()

        labelset = self._learn_labelset(raw_documents,
                                        self.fixed_labelset_)
        if not self.fixed_labelset_:
            self.labelset_ = labelset

        return self

    def fit_transform(self, raw_documents):
        """Learn the label encoder and return a vector of labels

        There is one label per instance extracted from raw_documents.
        """
        self._validate_labelset()

        labelset = self._learn_labelset(raw_documents,
                                        self.fixed_labelset_)
        if not self.fixed_labelset_:
            self.labelset_ = labelset
        # re-run through documents to generate y
        for lab in self._instance_labels(raw_documents):
            yield lab

    def transform(self, raw_documents):
        """Transform documents to a label vector"""
        if not hasattr(self, 'labelset_'):
            self._validate_labelset()
        if not self.labelset_:
            raise ValueError('Empty labelset')

        for lab in self._instance_labels(raw_documents):
            yield lab


# helper function to re-emit features from single EDUs in pairs
def re_emit(feats, suff):
    """Re-emit feats with suff appended to each feature name"""
    for fn, fv in feats:
        yield (fn + suff, fv)


class DocumentCountVectorizer(object):
    """Fancy vectorizer for the RST-DT treebank.

    See `sklearn.feature_extraction.text.CountVectorizer` for reference.
    """

    def __init__(self, instance_generator,
                 feature_set,
                 max_df=1.0, min_df=1, max_features=None,
                 vocabulary=None,
                 separator='='):
        """
        instance_generator to enumerate the instances from a doc
        feature_set is the feature set to use
        """
        # instance generator
        self.instance_generator = instance_generator
        # feature set
        self.feature_set = feature_set
        # EXPERIMENTAL
        # preprocessor for each EDU
        self.doc_preprocess = feature_set.build_doc_preprocessor()
        # feature extractor for single EDUs
        sing_header, sing_extract = feature_set.build_edu_feature_extractor()
        self.sing_header = sing_header
        self.sing_extract = sing_extract
        # feature extractor for pairs of EDUs
        pair_header, pair_extract = feature_set.build_pair_feature_extractor()
        self.pair_header = pair_header
        self.pair_extract = pair_extract
        # end EXPERIMENTAL
        # feature filters
        self.max_df = max_df
        self.min_df = min_df
        if max_df < 0 or min_df < 0:
            raise ValueError('negative value for max_df of min_df')
        self.max_features = max_features
        if max_features is not None:
            if ((not isinstance(max_features, numbers.Integral) or
                 max_features <= 0)):
                err_str = 'max_features={}, should be int > 0 or None'
                err_str = err_str.format(repr(max_features))
                raise ValueError(err_str)
        self.vocabulary = vocabulary
        # separator for one-hot-encoding
        self.separator = separator

    # document-level method
    def _extract_feature_vectors(self, doc):
        """Extract feature vectors for all EDU pairs of a document"""

        doc_preprocess = self.doc_preprocess
        sing_header = self.sing_header
        sing_extract = self.sing_extract
        pair_header = self.pair_header
        pair_extract = self.pair_extract
        separator = self.separator

        # preprocess each EDU
        edu_info = doc_preprocess(doc)

        # extract one feature vector per EDU pair
        feat_vecs = []
        # generate EDU pairs
        edu_pairs = self.instance_generator(doc)
        # cache single EDU features
        sf_cache = dict()

        for edu1, edu2 in edu_pairs:
            relevant_edus = {
                'EDU1': edu1,
                'EDU2': edu2,
            }

            if True:  # enable experimental contextual features
                # around e1
                e1_idx = edu1.num
                try:
                    relevant_edus['EDU1_m1'] = doc.edus[e1_idx - 1]
                except IndexError:
                    pass
                try:
                    relevant_edus['EDU1_p1'] = doc.edus[e1_idx + 1]
                except IndexError:
                    pass

                # around e2
                e2_idx = edu2.num
                try:
                    relevant_edus['EDU2_m1'] = doc.edus[e2_idx - 1]
                except IndexError:
                    pass
                try:
                    relevant_edus['EDU2_p1'] = doc.edus[e2_idx + 1]
                except IndexError:
                    pass

            # extract pair features
            feats = list(pair_extract(edu_info[edu1], edu_info[edu2]))

            # extract, cache and re-emit single features
            for ename, e in sorted(relevant_edus.items()):
                try:
                    efeats = sf_cache[e]
                except KeyError:
                    efeats = list(sing_extract(edu_info[e]))
                    sf_cache[e] = efeats

                suffix = '_{}'.format(ename)
                feats.extend(list(re_emit(efeats, suffix)))

            # apply one hot encoding for all string values
            oh_feats = []
            for f, v in feats:
                if isinstance(v, tuple):
                    f = '{}{}{}'.format(f, separator, str(v))
                    v = 1
                elif isinstance(v, (str, unicode)):
                    f = '{}{}{}'.format(f, separator, v)
                    v = 1
                oh_feats.append((f, v))
            # sum values of entries with same feature name
            feat_cnt = Counter()
            for fn, fv in oh_feats:
                feat_cnt[fn] += fv
            feat_vec = feat_cnt.items()  # non-deterministic order
            # could be : feat_vec = sorted(feat_cnt.items())
            feat_vecs.append(feat_vec)

        return feat_vecs

    # corpus level methods
    def _instances(self, raw_documents):
        """Extract instances, with only features that are in vocabulary"""
        vocabulary = self.vocabulary_

        analyze = self.build_analyzer()
        for doc in raw_documents:
            feat_vecs = analyze(doc)
            for feat_vec in feat_vecs:
                row = [(vocabulary[fn], fv)
                       for fn, fv in feat_vec
                       if fn in vocabulary]
                yield row

    def _vocab_df(self, raw_documents, fixed_vocab):
        """Gather vocabulary (if fixed_vocab=False) and doc frequency
        """
        if fixed_vocab:
            vocabulary = self.vocabulary_
        else:
            # add a new value when a new item is seen
            vocabulary = defaultdict()
            vocabulary.default_factory = vocabulary.__len__
        # track how many documents this feature appears in
        vocab_df = Counter()

        analyze = self.build_analyzer()
        for doc in raw_documents:
            feat_vecs = analyze(doc)
            doc_features = [fn for feat_vec in feat_vecs
                            for fn, fv in feat_vec]
            for feature in doc_features:
                try:
                    feat_id = vocabulary[feature]
                except KeyError:
                    # ignore out-of-vocabulary items for fixed_vocab=True
                    continue
            # second pass over doc features to update document frequency
            for feature in set(doc_features):
                if feature in vocabulary:
                    vocab_df[feature] += 1

        if not fixed_vocab:
            # disable defaultdict behaviour
            vocabulary = dict(vocabulary)
            if not vocabulary:
                raise ValueError('empty vocabulary')

        return vocabulary, vocab_df

    def _limit_vocabulary(self, vocabulary, vocab_df,
                          high=None, low=None, limit=None):
        """Remove too rare or too common features.

        Prune features that are non zero in more samples than high or less
        documents than low, restrict the vocabulary to at most the limit most
        frequent.

        Returns the set of removed features.
        """
        if high is None and low is None and limit is None:
            return set()

        # compute a mask based on vocab_df
        dfs = [vocab_df[feat] for feat, _ in sorted(vocabulary.items(),
                                                    key=lambda x: x[1])]
        mask = [1 for _ in dfs]
        if high is not None:
            mask = [m & (df <= high)
                    for m, df in itertools.izip(mask, dfs)]
        if low is not None:
            mask = [m & (df >= low)
                    for m, df in itertools.izip(mask, dfs)]
        if limit is not None:
            raise NotImplementedError('vocabulary cannot be limited... yet')

        # map old to new indices
        # pure python reimpl of np.cumsum(mask) - 1
        new_indices = []
        prev_idx = -1
        for m in mask:
            new_idx = prev_idx + m
            new_indices.append(new_idx)
            prev_idx = new_idx
        # removed features
        removed_feats = set()
        vocab_items = vocabulary.items()
        for feat, old_index in vocab_items:
            if mask[old_index]:
                vocabulary[feat] = new_indices[old_index]
            else:
                del vocabulary[feat]
                removed_feats.add(feat)

        return vocabulary, removed_feats

    def decode(self, doc):
        """Decode the input into a DocumentPlus

        doc is an educe.rst_dt.document_plus.DocumentPlus
        """
        if not isinstance(doc, DocumentPlus):
            # doc = self.decoder(doc)
            raise ValueError('doc should be a DocumentPlus')
        return doc

    def build_analyzer(self):
        """Return a callable that extracts feature vectors from a doc"""
        return lambda doc: self._extract_feature_vectors(self.decode(doc))

    def _validate_vocabulary(self):
        """Validate vocabulary"""
        vocabulary = self.vocabulary
        if vocabulary is not None:
            if not vocabulary:
                raise ValueError('empty vocabulary passed to fit')
            self.fixed_vocabulary_ = True
            self.vocabulary_ = dict(vocabulary)
        else:
            self.fixed_vocabulary_ = False

    def fit(self, raw_documents, y=None):
        """Learn a vocabulary dictionary of all features from the documents"""
        self.fit_transform(raw_documents)
        return self

    def fit_transform(self, raw_documents, y=None):
        """Learn the vocabulary dictionary and generate (row, (tgt, src))
        """
        self._validate_vocabulary()
        max_df = self.max_df
        min_df = self.min_df
        max_features = self.max_features

        vocabulary, vocab_df = self._vocab_df(raw_documents,
                                              self.fixed_vocabulary_)

        if not self.fixed_vocabulary_:
            n_doc = len(raw_documents)
            max_doc_count = (max_df
                             if isinstance(max_df, numbers.Integral)
                             else max_df * n_doc)
            min_doc_count = (min_df
                             if isinstance(min_df, numbers.Integral)
                             else min_df * n_doc)
            if max_doc_count < min_doc_count:
                raise ValueError(
                    'max_df corresponds to < documents than min_df')
            # limit features with df
            vocabulary, rm_feats = self._limit_vocabulary(vocabulary,
                                                          vocab_df,
                                                          high=max_doc_count,
                                                          low=min_doc_count,
                                                          limit=max_features)
            self.vocabulary_ = vocabulary
        # re-run through documents to generate X
        for row in self._instances(raw_documents):
            yield row

    def transform(self, raw_documents):
        """Transform documents to a feature matrix

        Note: generator of (row, (tgt, src))
        """
        if not hasattr(self, 'vocabulary_'):
            self._validate_vocabulary()
        if not self.vocabulary_:
            raise ValueError('Empty vocabulary')

        for row in self._instances(raw_documents):
            yield row
