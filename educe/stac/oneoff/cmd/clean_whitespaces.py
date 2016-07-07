"""Fix whitespaces in the glozz .ac files.

This script currently adds missing initial and final whitespaces in
glozz .ac files and ensures that neither is covered by an annotation.

In the near future, it should also delete excess whitespaces in the middle
of the text that were created by buggy versions of `stac-edit move`.
"""

from __future__ import print_function

from educe.annotation import Span
from educe.stac.util.args import (add_usual_output_args,
                                  announce_output_dir,
                                  get_output_dir,
                                  read_corpus)
from educe.stac.util.doc import (delete_text_at_span,
                                 evil_set_text,
                                 shift_annotations)
from educe.stac.util.output import save_document

NAME = 'clean-whitespaces'


def config_argparser(parser):
    """Subcommand flags.

    You should create and pass in the subparser to which the flags are
    to be added.
    """
    parser.add_argument('corpus', metavar='DIR',
                        nargs='?',
                        help='corpus dir')
    add_usual_output_args(parser, default_overwrite=True)
    parser.set_defaults(func=main)


def fix_surrounding_whitespaces(doc):
    """Ensure that a document text starts and ends with a whitespace.

    This checks for a leading and trailing whitespace and fixes them
    if necessary.
    The leading and trailing whitespaces are restored such that they
    do not belong to any annotation.

    This is still work-in-progress.

    Parameters
    ----------
    doc: Document
        Original document

    Returns
    -------
    doc: Document
        Updated document
    """
    # ensure there is exactly one leading whitespace:
    # remove all present, prepend exactly one
    sp_left = len(doc.text()) - len(doc.text().lstrip())
    if sp_left:
        doc, _ = delete_text_at_span(doc, Span(0, sp_left))
    doc = shift_annotations(doc, 1)

    # ensure there is exactly one trailing whitespace:
    # remove all present, append exactly one
    doc_txt_len = len(doc.text())
    sp_right = len(doc.text()) - len(doc.text().rstrip())
    if sp_right:
        doc, _ = delete_text_at_span(doc, Span(doc_txt_len - sp_right,
                                               doc_txt_len))
    evil_set_text(doc, doc.text() + ' ')

    return doc


def main(args):
    """Subcommand main.

    You shouldn't need to call this yourself if you're using
    `config_argparser`.
    """
    corpus = read_corpus(args)
    output_dir = get_output_dir(args, default_overwrite=True)
    for k, doc in corpus.items():
        doc = fix_surrounding_whitespaces(doc)
        save_document(output_dir, k, doc)
    announce_output_dir(output_dir)
