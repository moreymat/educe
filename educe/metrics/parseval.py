"""Parseval metrics for constituency trees.

TODO
----
* [ ] factor out the report from the parseval scoring function, see
`sklearn.metrics.classification.classification_report`
* [ ] refactor the selection functions that enable to break down
evaluations, to avoid almost duplicates (as currently)
"""

from __future__ import absolute_import, print_function

import numpy as np

from educe.metrics.scores_structured import (precision_recall_fscore_support,
                                             unique_labels)


def parseval_scores(ctree_true, ctree_pred, subtree_filter=None,
                    exclude_root=False, lbl_fn=None, labels=None,
                    average=None, per_doc=False):
    """Compute PARSEVAL scores for ctree_pred wrt ctree_true.

    Parameters
    ----------
    ctree_true : list of list of RSTTree or SimpleRstTree
        List of reference RST trees, one per document.

    ctree_pred : list of list of RSTTree or SimpleRstTree
        List of predicted RST trees, one per document.

    subtree_filter : function, optional
        Function to filter all local trees.

    exclude_root : boolean, defaults to True
        If True, exclude the root node of both ctrees from the eval.

    lbl_fn: function, optional
        Function to relabel spans.

    labels : list of string, optional
        Corresponds to sklearn's target_names IMO

    average : one of {'micro', 'macro'}, optional
        TODO, see scores_structured

    per_doc : boolean, optional
        If True, precision, recall and f1 are computed for each document
        separately then averaged over documents.
        (TODO this should probably be pushed down to
        `scores_structured.precision_recall_fscore_support`)

    Returns
    -------
    precision : float (if average is not None) or array of float, shape =\
        [n_unique_labels]
        Weighted average of the precision of each class.

    recall : float (if average is not None) or array of float, shape =\
        [n_unique_labels]

    fbeta_score : float (if average is not None) or array of float, shape =\
        [n_unique_labels]

    support_true : int (if average is not None) or array of int, shape =\
        [n_unique_labels]
        The number of occurrences of each label in ``ctree_true``.

    support_pred : int (if average is not None) or array of int, shape =\
        [n_unique_labels]
        The number of occurrences of each label in ``ctree_pred``.

    """
    # extract descriptions of spans from the true and pred trees
    spans_true = [ct.get_spans(subtree_filter=subtree_filter,
                               exclude_root=exclude_root)
                  for ct in ctree_true]
    spans_pred = [ct.get_spans(subtree_filter=subtree_filter,
                               exclude_root=exclude_root)
                  for ct in ctree_pred]
    # use lbl_fn to define labels
    if lbl_fn is not None:
        spans_true = [[(span[0], lbl_fn(span)) for span in spans]
                      for spans in spans_true]
        spans_pred = [[(span[0], lbl_fn(span)) for span in spans]
                      for spans in spans_pred]

    # NEW gather present labels
    present_labels = unique_labels(spans_true, spans_pred)
    if labels is None:
        labels = present_labels
    else:
        # currently not tested
        labels = np.hstack([labels, np.setdiff1d(present_labels, labels,
                                                 assume_unique=True)])
    # end NEW labels

    if per_doc:
        # non-standard variant that computes scores per doc then
        # averages them over docs ; this variant is implemented in DPLP
        # where it is mistaken for the standard version
        scores = []
        for doc_spans_true, doc_spans_pred in zip(spans_true, spans_pred):
            p, r, f1, s_true, s_pred = precision_recall_fscore_support(
                [doc_spans_true], [doc_spans_pred], labels=labels,
                average=average)
            scores.append((p, r, f1, s_true, s_pred))
        p, r, f1, s_true, s_pred = (
            np.array([x[0] for x in scores]).mean(),
            np.array([x[1] for x in scores]).mean(),
            np.array([x[2] for x in scores]).mean(),
            np.array([x[3] for x in scores]).sum(),
            np.array([x[4] for x in scores]).sum()
        )
    else:
        # standard version of this eval
        p, r, f1, s_true, s_pred = precision_recall_fscore_support(
            spans_true, spans_pred, labels=labels, average=average)

    return p, r, f1, s_true, s_pred, labels


def parseval_report(ctree_true, ctree_pred, exclude_root=False,
                    subtree_filter=None, lbl_fns=None, digits=4,
                    print_support_pred=True, per_doc=False):
    """Build a text report showing the PARSEVAL discourse metrics.

    This is the simplest report we need to generate, it corresponds
    to the arrays of results from the literature.
    Metrics are calculated globally (average='micro').

    Parameters
    ----------
    ctree_true: TODO
        TODO
    ctree_pred: TODO
        TODO
    metric_types: list of strings, optional
        Metrics that need to be included in the report ; if None is
        given, defaults to ['S', 'S+N', 'S+R', 'S+N+R'].
    digits: int, defaults to 4
        Number of decimals to print.
    print_support_pred: boolean, defaults to True
        If True, the predicted support, i.e. the number of predicted
        spans, is also displayed. This is useful for non-binary ctrees
        as the number of spans in _true and _pred can differ.
    span_sel: TODO
        TODO
    per_doc: boolean, defaults to False
        If True, compute p, r, f for each doc separately then compute the
        mean of each score over docs. This is *not* the correct
        implementation, but it corresponds to that in DPLP.
    """
    if lbl_fns is None:
        # we require a labelled span to be a pair (span, lbl)
        # where span and lbl can be anything, for example
        # * span = (span_beg, span_end)
        # * lbl = (nuc, rel)
        lbl_fns = [('Labelled Span', lambda span_lbl: span_lbl[1])]

    metric_types = [k for k, v in lbl_fns]

    # prepare scaffold for report
    width = max(len(str(x)) for x in metric_types)
    width = max(width, digits)
    headers = ["precision", "recall", "f1-score", "support", "sup_pred"]
    fmt = '%% %ds' % width  # first col: class name
    fmt += '  '
    fmt += ' '.join(['% 9s' for _ in headers])
    fmt += '\n'
    headers = [""] + headers
    report = fmt % tuple(headers)
    report += '\n'

    # compute scores
    metric_scores = dict()
    for metric_type, lbl_fn in lbl_fns:
        p, r, f1, s_true, s_pred, labels = parseval_scores(
            ctree_true, ctree_pred, subtree_filter=subtree_filter,
            exclude_root=exclude_root, lbl_fn=lbl_fn, labels=None,
            average='micro', per_doc=per_doc)
        metric_scores[metric_type] = (p, r, f1, s_true, s_pred)

    # fill report
    for metric_type in metric_types:
        (p, r, f1, s_true, s_pred) = metric_scores[metric_type]
        values = [metric_type]
        for v in (p, r, f1):
            values += ["{0:0.{1}f}".format(v, digits)]
        values += ["{0}".format(s_true)]  # support_true
        values += ["{0}".format(s_pred)]  # support_pred
        report += fmt % tuple(values)

    return report


def parseval_detailed_report(ctree_true, ctree_pred, exclude_root=False,
                             subtree_filter=None, lbl_fn=None,
                             labels=None, sort_by_support=True,
                             digits=4, per_doc=False):
    """Build a text report showing the PARSEVAL discourse metrics.

    FIXME model after sklearn.metrics.classification.classification_report

    Parameters
    ----------
    ctree_true : list of RSTTree or SimpleRstTree
        Ground truth (correct) target structures.

    ctree_pred : list of RSTTree or SimpleRstTree
        Estimated target structures as predicted by a parser.

    labels : list of string, optional
        Relation labels to include in the evaluation.
        FIXME Corresponds more to target_names in sklearn IMHO.

    lbl_fn : function from tuple((int, int), (string, string)) to string
        Label extraction function

    digits : int
        Number of digits for formatting output floating point values.

    Returns
    -------
    report : string
        Text summary of the precision, recall, F1 score, support for each
        class (or micro-averaged over all classes).

    """
    if lbl_fn is None:
        # we require a labelled span to be a pair (span, lbl)
        # where span and lbl can be anything, for example
        # * span = (span_beg, span_end)
        # * lbl = (nuc, rel)
        lbl_fn = ('Labelled Span', lambda span_lbl: span_lbl[1])

    # call with average=None to compute per-class scores, then
    # compute average here and print it
    p, r, f1, s_true, s_pred, labels = parseval_scores(
        ctree_true, ctree_pred, subtree_filter=subtree_filter,
        exclude_root=exclude_root, lbl_fn=lbl_fn, labels=labels,
        average=None, per_doc=per_doc)

    # scaffold for report
    last_line_heading = 'avg / total'

    width = max(len(str(lbl)) for lbl in labels)
    width = max(width, len(last_line_heading), digits)

    headers = ["precision", "recall", "f1-score", "support", "sup_pred"]
    fmt = '%% %ds' % width  # first col: class name
    fmt += '  '
    fmt += ' '.join(['% 9s' for _ in headers])
    fmt += '\n'

    headers = [""] + headers
    report = fmt % tuple(headers)
    report += '\n'

    # map labels to indices, possibly sorted by their support
    sorted_ilbls = enumerate(labels)
    if sort_by_support:
        sorted_ilbls = sorted(sorted_ilbls, key=lambda x: s_true[x[0]],
                              reverse=True)
    # one line per label
    for i, label in sorted_ilbls:
        values = [label]
        for v in (p[i], r[i], f1[i]):
            values += ["{0:0.{1}f}".format(v, digits)]
        values += ["{0}".format(s_true[i])]
        values += ["{0}".format(s_pred[i])]
        if average is None:
            # print per-class scores for average=None only
            report += fmt % tuple(values)

    if average is None:
        # print only if per-class scores
        report += '\n'

    # last line ; compute averages
    values = [last_line_heading]
    for v in (np.average(p, weights=s_true),
              np.average(r, weights=s_true),
              np.average(f1, weights=s_true)):
        values += ["{0:0.{1}f}".format(v, digits)]
    values += ['{0}'.format(np.sum(s_true))]
    values += ['{0}'.format(np.sum(s_pred))]
    report += fmt % tuple(values)

    return report
