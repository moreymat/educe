# Author: Eric Kow
# License: CeCILL-B (French BSD3-like)

"""
Merge several dialogue level annotations into one
"""

from __future__ import print_function
import copy
import sys

from educe.annotation import Span
from educe.glozz import GlozzException

from educe.stac.annotation import TurnId
from educe.stac.util.annotate import annotate_doc
from educe.stac.util.args import\
    add_usual_input_args, add_usual_output_args, anno_id,\
    add_commit_args,\
    read_corpus,\
    get_output_dir, announce_output_dir
from educe.stac.util.glozz import\
    anno_id_from_tuple, anno_id_to_tuple,\
    get_turn, is_dialogue
from educe.stac.util.output import save_document


def _get_annotation_with_id(sought_tuple, annotations):
    """
    Given a tuple (author,creation_date), pick out the one annotation
    whose id matches.  There must be exactly one.
    """
    sought = anno_id_from_tuple(sought_tuple)
    candidates = [x for x in annotations if x.local_id() == sought]
    if len(candidates) == 1:
        return candidates[0]
    elif len(candidates) > 1:
        raise Exception('More than one annotation found with id %s' % sought)
    else:
        raise Exception('No annotations found with id %s' % sought)


def _concatenate_features(annotations, feature):
    """
    Concatenate the values for the given features for all the annotations.
    Ignore cases where the feature is unset
    """
    values_ = [x.features.get(feature) for x in annotations]
    values = [x for x in values_ if x]
    if values:
        return " ".join(values)
    else:
        return None


# TODO: this could be rewritten using the enclosure graph
def _dialogues_in_turns(corpus, turn1, turn2):
    """
    Given a pair of turns
    """

    # grab a document from the set (assumption here is that
    # they are all morally the same doc)
    if not corpus.values():
        sys.exit("No documents selected")
    doc = corpus.values()[0]

    starting_turn = get_turn(turn1, doc)
    ending_turn = get_turn(turn2, doc)

    # there's a bit of fuzz for whitespace before/after the
    # turns
    span = Span(starting_turn.text_span().char_start - 1,
                ending_turn.text_span().char_end + 1)

    def is_in_range(anno):
        """
        If the annotation is a dialogue that is covered by the
        turns in question
        """
        return is_dialogue(anno) and span.encloses(anno.span)

    return [anno_id_to_tuple(x.local_id()) for x in doc.annotations()
            if is_in_range(x)]


def _merge_spans(annos):
    """
    Return one big span stretching across all the annotations
    """
    return Span(min(x.text_span().char_start for x in annos),
                max(x.text_span().char_end for x in annos))


def _merge_dialogues_in_document(sought, doc):
    """
    Given an iterable of dialogue annotation ids, merge them
    replace the relevant dialogue annotations with a single
    combined one. Take the id (creation date, author,etc)
    from the first of the dialogues.

    NB: modifies the document
    """
    dialogues_ = [_get_annotation_with_id(d, doc.units) for d in sought]
    dialogues = sorted(dialogues_,
                       key=lambda x: x.text_span().char_start)
    combined = copy.deepcopy(dialogues[0])
    combined.span = _merge_spans(dialogues)
    for feat in ['Trades', 'Gets', 'Dice_rolling']:
        combined.features[feat] = _concatenate_features(dialogues, feat)

    # in-place replacement
    for i, _ in enumerate(doc.units):
        if doc.units[i] in dialogues:
            dialogues.remove(doc.units[i])
            doc.units[i] = combined
            break
    for dialogue in dialogues:
        doc.units.remove(dialogue)


# ---------------------------------------------------------------------
# command and args
# ---------------------------------------------------------------------

NAME = 'merge-dialogue'


def config_argparser(parser):
    """
    Subcommand flags.

    You should create and pass in the subparser to which the flags
    are to be added.
    """
    add_usual_input_args(parser, doc_subdoc_required=True,
                         help_suffix='in which to merge')
    add_usual_output_args(parser, default_overwrite=True)
    parser_mutex = parser.add_mutually_exclusive_group(required=True)
    parser_mutex.add_argument('--dialogues',
                              metavar='ANNO_ID', type=anno_id,
                              nargs='+',
                              help='eg. stac_39819045 stac_98871771')
    parser_mutex.add_argument('--turns',
                              metavar='TURN_ID', type=TurnId.from_string,
                              nargs=2,
                              help='eg. 187 192')
    add_commit_args(parser)
    parser.set_defaults(func=main)


def commit_msg(args, corpus, k, sought):
    """
    Generate a commit message describing the dialogue merging operation
    we are about to do (has to be run before merging happens)
    """
    doc = corpus[k]
    dstr = ", ".join(anno_id_from_tuple(x) for x in sought)
    dialogues = [_get_annotation_with_id(d, doc.units) for d in sought]
    if dialogues:
        title_fmt = u"{doc}_{subdoc}: merge dialogues{hint}"
        title_hint = (" (turns {}-{})".format(args.turns[0], args.turns[1])
                      if args.turns else "")
        dspan = _merge_spans(dialogues)
        lines = [title_fmt.format(doc=k.doc,
                                  subdoc=k.subdoc,
                                  hint=title_hint),
                 "",
                 "Dialogues ({}), was:".format(dstr),
                 "",
                 annotate_doc(doc, span=dspan)]
        return "\n".join(lines)
    else:
        return "(no commit message; nothing to merge)"


def main(args):
    """
    Subcommand main.

    You shouldn't need to call this yourself if you're using
    `config_argparser`
    """

    if not args.turns and len(args.dialogues) < 2:
        sys.exit("Must specify at least two dialogues")
    output_dir = get_output_dir(args, default_overwrite=True)
    corpus = read_corpus(args, verbose=True)
    if args.turns:
        try:
            sought = _dialogues_in_turns(corpus, args.turns[0], args.turns[1])
            if len(sought) < 2:
                sys.exit("Must specify at least two dialogues")
            print("Merging dialogues: " +
                  ", ".join(anno_id_from_tuple(x) for x in sought),
                  file=sys.stderr)
        except GlozzException as oops:
            sys.exit(str(oops))
    else:
        sought = args.dialogues
    if corpus and not args.no_commit_msg:
        key0 = list(corpus)[0]
        # compute this before we change things
        cmsg = commit_msg(args, corpus, key0, sought)
    for k in corpus:
        doc = corpus[k]
        _merge_dialogues_in_document(sought, doc)
        save_document(output_dir, k, doc)
    announce_output_dir(output_dir)
    if corpus and not args.no_commit_msg:
        print("-----8<------")
        print(cmsg)
