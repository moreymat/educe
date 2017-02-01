"""This submodule implements document vectorizers"""

from __future__ import print_function

import itertools
import numbers

from collections import defaultdict, Counter

from educe.rst_dt.document_plus import DocumentPlus


class DocumentLabelExtractor(object):
    """Label extractor for the RST-DT treebank.

    Parameters
    ----------
    instance_generator : generator
        Generator that enumerates the instances from a doc.

    ordered_pairs : boolean (default: True)
        True if the generated instances are ordered pairs of DUs:
        (du1, du2) != (du2, du1).

    unknown_label : str
        Reserved label for unknown cases.

    labelset : TODO
        TODO

    Attributes
    ----------
    fixed_labelset_ : boolean
        True if the labelset has been fixed, i.e. `self` has been fit.

    labelset_ : dict
        A mapping of labels to indices.

    """

    def __init__(self, instance_generator,
                 ordered_pairs=True,
                 unknown_label='__UNK__',
                 labelset=None):
        self.instance_generator = instance_generator
        self.ordered_pairs = ordered_pairs  # 2016-09-30
        self.unknown_label = unknown_label
        self.labelset = labelset

    def _extract_labels(self, doc):
        """Extract a label for each EDU pair extracted from `doc`.

        Parameters
        ----------
        doc : DocumentPlus
            Rich representation of the document.

        Returns
        -------
        labels : list of strings or None
            List of labels, one for every pair of EDUs (in the order
            in which they are generated by `self.instance_generator()`).
        """
        edu_pairs = self.instance_generator(doc)
        # extract one label per EDU pair
        # WIP 2016-09-30: ordered
        labels = doc.relations(edu_pairs, ordered=self.ordered_pairs)
        return labels

    def _instance_labels(self, raw_documents):
        """Extract label of instances, restricted to labelset"""
        labelset = self.labelset_
        # unknown labels
        unk_lab_id = labelset[self.unknown_label]

        analyze = self.build_analyzer()
        for doc in raw_documents:
            doc_labels = analyze(doc)
            yield [labelset.get(lab, unk_lab_id) for lab in doc_labels]

    def _learn_labelset(self, raw_documents, fixed_labelset):
        """Learn the labelset"""
        if fixed_labelset:
            labelset = self.labelset_
        else:
            # add a new value when a new label is seen
            labelset = defaultdict()
            labelset.default_factory = labelset.__len__
        # the id of the unknown label should be 0
        unk_lab = self.unknown_label
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
        """Currently a no-op if doc is a DocumentPlus.

        Raises an exception otherwise.
        Was: Decode the input into a DocumentPlus.

        Parameters
        ----------
        doc: DocumentPlus
            Rich representation of the document.

        Returns
        -------
        doc: DocumentPlus
            Rich representation of `doc`.
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
        for doc_labs in self._instance_labels(raw_documents):
            yield doc_labs

    def transform(self, raw_documents):
        """Transform documents to a label vector"""
        if not hasattr(self, 'labelset_'):
            self._validate_labelset()
        if not self.labelset_:
            raise ValueError('Empty labelset')

        for doc_labs in self._instance_labels(raw_documents):
            yield doc_labs


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
                 lecsie_data_dir=None,
                 max_df=1.0, min_df=1, max_features=None,
                 vocabulary=None,
                 separator='=',
                 split_feat_space=None):
        """
        Parameters
        ----------
        instance_generator: generator(instances)
            generator to enumerate the instances from a doc
        feature_set: class
            which feature set to use
        lecsie_data_dir: string
            Path to the directory containing LECSIE feature files
        split_feat_space: string, optional
            If not None, indicates the features on which the feature space
            should be split. Possible values are 'dir', 'sent', 'dir_sent'.
        """
        # instance generator
        self.instance_generator = instance_generator
        # feature set
        self.feature_set = feature_set
        # EXPERIMENTAL
        # preprocessor for each EDU
        self.doc_preprocess = feature_set.build_doc_preprocessor()
        # feature extractor for single EDUs
        sing_extract = feature_set.build_edu_feature_extractor()
        self.sing_extract = sing_extract
        # feature extractor for pairs of EDUs
        pair_extract = feature_set.build_pair_feature_extractor(
            lecsie_data_dir=lecsie_data_dir)
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
        # NEW whether to split the feature space
        self.split_feat_space = split_feat_space

    # document-level method
    def _extract_feature_vectors(self, doc):
        """Extract feature vectors for all EDU pairs of a document.

        Parameters
        ----------
        doc: educe.rst_dt.document_plus.DocumentPlus
            Rich representation of the document.

        Returns
        -------
        feat_vecs: list of feature vectors
            List of feature vectors, one for every pair of EDUs (in the
            order in which they are generated by
            `self.instance_generator()`).

        Notes
        -----
        This is a bottleneck for speed.
        """

        doc_preprocess = self.doc_preprocess
        sing_extract = self.sing_extract
        pair_extract = self.pair_extract
        separator = self.separator
        # NEW
        feat_prod = self.feature_set.product_features
        feat_comb = self.feature_set.combine_features
        # NEW 2
        split_feat_space = self.split_feat_space
        # end NEW

        edu2para = doc.edu2para
        # preprocess each EDU
        edu_infos, para_infos = doc_preprocess(doc)

        # extract one feature vector per EDU pair
        feat_vecs = []
        # generate EDU pairs
        edu_pairs = self.instance_generator(doc)
        # cache single EDU features
        sf_cache = dict()

        for edu1, edu2 in edu_pairs:
            edu1_num = edu1.num
            edu2_num = edu2.num
            # WIP interval
            if edu1_num < edu2_num:
                edul_num = edu1_num
                edur_num = edu2_num
            else:
                edul_num = edu2_num
                edur_num = edu1_num
            bwn_nums = range(edul_num + 1, edur_num)
            # end WIP interval

            feat_dict = dict()
            # retrieve info for each EDU
            edu_info1 = edu_infos[edu1_num]
            edu_info2 = edu_infos[edu2_num]
            # NEW paragraph info
            try:
                para_info1 = para_infos[edu2para[edu1_num]]
            except TypeError:
                para_info1 = None
            try:
                para_info2 = para_infos[edu2para[edu2_num]]
            except TypeError:
                para_info2 = None
            # ... and for the EDUs in between (WIP interval)
            edu_info_bwn = [edu_infos[i] for i in bwn_nums]

            # gov EDU
            if edu1_num not in sf_cache:
                sf_cache[edu1_num] = dict(sing_extract(
                    doc, edu_info1, para_info1))
            feat_dict['EDU1'] = dict(sf_cache[edu1_num])
            # dep EDU
            if edu2_num not in sf_cache:
                sf_cache[edu2_num] = dict(sing_extract(
                    doc, edu_info2, para_info2))
            feat_dict['EDU2'] = dict(sf_cache[edu2_num])
            # pair + in between
            feat_dict['pair'] = dict(pair_extract(
                doc, edu_info1, edu_info2, edu_info_bwn))
            # NEW
            # product features
            feat_dict['pair'].update(feat_prod(feat_dict['EDU1'],
                                               feat_dict['EDU2'],
                                               feat_dict['pair']))
            # combine features
            feat_dict['pair'].update(feat_comb(feat_dict['EDU1'],
                                               feat_dict['EDU2'],
                                               feat_dict['pair']))
            # add suffix to single EDU features
            feat_dict['EDU1'] = dict(re_emit(feat_dict['EDU1'].items(),
                                             '_EDU1'))
            feat_dict['EDU2'] = dict(re_emit(feat_dict['EDU2'].items(),
                                             '_EDU2'))

            # split feat space
            if split_feat_space is not None:
                # options are:
                # * directionality of attachment
                # * intra/inter-sentential,
                # * intra/inter-sentential + attachment dir
                fds = self.feature_set.split_feature_space(
                    feat_dict['EDU1'],
                    feat_dict['EDU2'],
                    feat_dict['pair'],
                    keep_original=False,
                    split_criterion=split_feat_space)
                feat_dict['EDU1'], feat_dict['EDU2'], feat_dict['pair'] = fds

            # convert to list
            feats = list(itertools.chain.from_iterable(
                fd.items() for fd in feat_dict.values()))
            # end NEW

            # apply one hot encoding for all string values
            oh_feats = []
            for f, v in feats:
                if isinstance(v, tuple):
                    f = '{}{}{}'.format(f, separator, str(v))
                    v = 1
                elif isinstance(v, (str, unicode)):
                    # NEW explicitly replace with regular spaces the
                    # non-breaking spaces that appear in CoreNLP output
                    # for fractions of a dollar in stock prices,
                    # e.g. "100 3/32" ;
                    # non-breaking spaces might appear elsewhere ;
                    # svmlight format expects ascii characters so it makes
                    # some sense to replace and convert to ascii here
                    if isinstance(v, unicode):
                        v2 = v.replace(u'\xa0', u' ')
                        v = v2.encode('utf-8')
                    # end NEW
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
            doc_rows = [[(vocabulary[fn], fv) for fn, fv in feat_vec
                         if fn in vocabulary]
                        for feat_vec in feat_vecs]
            yield doc_rows

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

    def _limit_features(self, vocab_df, vocabulary, high=None, low=None,
                        limit=None):
        """Remove too rare or too common features.

        Prune features that are non zero in more samples than high or less
        documents than low, modifying the vocabulary and restricting it to
        (TODO at most the limit most frequent).

        This does not prune samples with zero features.

        This is essentially a reimplementation of the one in
        sklearn.feature_extraction.text.CountVectorizer, except vocab_df
        is computed differently.
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
        vocab_items = vocabulary.items()
        for feat, old_index in vocab_items:
            if mask[old_index]:
                vocabulary[feat] = new_indices[old_index]
            else:
                del vocabulary[feat]

        return vocabulary

    def decode(self, doc):
        """Decode the input into a DocumentPlus.

        Currently a no-op except for type checking.

        Parameters
        ----------
        doc: educe.rst_dt.document_plus.DocumentPlus
            Rich representation of the document.

        Returns
        -------
        doc: educe.rst_dt.document_plus.DocumentPlus
            Rich representation of the document.
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
            vocabulary = self._limit_features(vocab_df, vocabulary,
                                              high=max_doc_count,
                                              low=min_doc_count,
                                              limit=max_features)
            self.vocabulary_ = vocabulary
        return self

    def fit_transform(self, raw_documents, y=None):
        """Learn the vocabulary dictionary and generate a feature matrix per document.
        """
        self.fit(raw_documents, y=y)
        return self.transform(raw_documents)

    def transform(self, raw_documents):
        """Transform each document to a feature matrix.

        Generate a feature matrix (one row per instance) for each document.

        Parameters
        ----------
        raw_documents : TODO
            TODO

        Yields
        ------
        feat_matrix : (row, (tgt, src))
            Feature matrix for the next document.
        """
        if not hasattr(self, 'vocabulary_'):
            self._validate_vocabulary()
        if not self.vocabulary_:
            raise ValueError('Empty vocabulary')

        for feat_matrix in self._instances(raw_documents):
            yield feat_matrix
