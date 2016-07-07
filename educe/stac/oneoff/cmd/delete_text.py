"""Delete a text span in a document.

"""

from __future__ import print_function
import sys

import educe.stac
from educe.stac.edit.cmd.move import is_requested
from educe.stac.util.annotate import annotate_doc, show_diff
from educe.stac.util.args import (add_commit_args,
                                  add_usual_input_args,
                                  add_usual_output_args,
                                  announce_output_dir,
                                  comma_span,
                                  get_output_dir)
from educe.stac.util.doc import delete_text_at_span
from educe.stac.util.output import save_document


NAME = 'delete-text'


def commit_msg(key, anno_str_before, all_del_annos):
    """Generate a commit message describing the operation we just did.

    """
    lines = [
        "{}_{}: very scary edit (delete text)".format(
            key.doc, key.subdoc),
        "",
        "    " + anno_str_before,
        "==> " + '',
        ""
    ]

    if all_del_annos:
        lines.append("======= Deleted annotations =======")
    for tgt_k, del_annos in sorted(all_del_annos):
        lines.append(
            '------- {}{} -------'.format(
                tgt_k.stage,
                (' / ' + tgt_k.annotator if tgt_k.annotator is not None
                 else ''))
        )
        if del_annos[0]:
            lines.append(
                'Units: ' + ', '.join(sorted(
                    str(x.local_id()) for x in del_annos[0]))
            )
        if del_annos[1]:
            lines.append(
                'Schemas: ' + ', '.join(sorted(
                    str(x.local_id()) for x in del_annos[1]))
            )
        if del_annos[2]:
            lines.append(
                'Relations: ' + ', '.join(sorted(
                    str(x.local_id()) for x in del_annos[2]))
            )

    return "\n".join(lines)


def config_argparser(parser):
    """Subcommand flags.

    You should create and pass in the subparser to which the flags
    are to be added.
    """
    add_usual_input_args(parser, doc_subdoc_required=True,
                         help_suffix='to insert into')
    parser.add_argument('--span', metavar='SPAN', type=comma_span,
                        required=True,
                        help='span of the substitution site')
    parser.add_argument('--minor', action='store_true',
                        help='minor fix, leave annotations as they are')
    add_usual_output_args(parser, default_overwrite=True)
    add_commit_args(parser)
    parser.set_defaults(func=main)


def main(args):
    """Subcommand main.

    You shouldn't need to call this yourself if you're using
    `config_argparser`.
    """
    output_dir = get_output_dir(args, default_overwrite=True)

    # locate insertion site: target document
    reader = educe.stac.Reader(args.corpus)
    tgt_files = reader.filter(reader.files(), is_requested(args))
    tgt_corpus = reader.slurp(tgt_files)

    # TODO mark units with FIXME, optionally delete in/out relations
    span = args.span
    minor = args.minor
    # store before/after
    annos_before = []
    all_del_annos = []
    for tgt_k, tgt_doc in tgt_corpus.items():
        annos_before.append(annotate_doc(tgt_doc, span=span))
        # process
        new_tgt_doc, del_annos = delete_text_at_span(
            tgt_doc, span, minor=minor)
        all_del_annos.append((tgt_k, del_annos))
        # show diff and save doc
        diffs = ["======= DELETE TEXT IN %s   ========" % tgt_k,
                 show_diff(tgt_doc, new_tgt_doc)]
        print("\n".join(diffs).encode('utf-8'), file=sys.stderr)
        save_document(output_dir, tgt_k, new_tgt_doc)
    announce_output_dir(output_dir)
    # commit message
    tgt_k, tgt_doc = list(tgt_corpus.items())[0]
    anno_str_before = annos_before[0]
    if tgt_k and not args.no_commit_msg:
        print("-----8<------")
        print(commit_msg(tgt_k, anno_str_before, all_del_annos))
