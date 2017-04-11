"""This submodule implements document vectorizers"""

# pylint: disable=too-few-public-methods

from __future__ import absolute_import, print_function

from collections import defaultdict, namedtuple, Sequence
import itertools
import copy
import os
import sys

from nltk.corpus import verbnet as vnet

from educe.learning.keys import KeyGroup
from educe.stac import postag, corenlp
from educe.stac.annotation import addressees, speaker, is_relation_instance
from educe.stac.corpus import twin_key
from educe.stac.fusion import Dialogue, FakeRootEDU, fuse_edus
from educe.stac.learning.features import (clean_dialogue_act, SingleEduKeys,
                                          PairKeys)
# import educe.stac.lexicon.inquirer as inquirer
import educe.stac.lexicon.pdtb_markers as pdtb_markers
from educe.stac.lexicon.wordclass import Lexicon
import educe.stac.graph as stac_gr
import educe.util
import educe.stac


UNK = '__UNK__'
ROOT = 'ROOT'
UNRELATED = 'UNRELATED'


class DialogueActVectorizer(object):
    """Dialogue act extractor for the STAC corpus."""

    def __init__(self, instance_generator, labels):
        """
        instance_generator to enumerate the instances from a doc

        :type labels: list(string)
        """
        self.instance_generator = instance_generator
        self.labelset_ = {l: i for i, l in enumerate(labels, start=1)}
        self.labelset_[UNK] = 0

    def transform(self, raw_documents):
        """Learn the label encoder and return a vector of labels

        There is one label per instance extracted from raw_documents.

        Parameters
        ----------
        raw_documents : list of `educe.stac.fusion.Dialogue`
            List of dialogues.

        Yields
        ------
        inst_lbls : list of int
            (Integer) label for each instance of the next document.
        """
        # run through documents to generate y
        for doc in raw_documents:
            inst_lbls = []
            for edu in self.instance_generator(doc):
                label = clean_dialogue_act(edu.dialogue_act() or UNK)
                inst_lbls.append(self.labelset_[label])
            yield inst_lbls


class LabelVectorizer(object):
    """Label extractor for the STAC corpus.

    Parameters
    ----------
    instance_generator : fun(doc) -> :obj:`list` of (EDU, EDU)
        Function to enumerate the candidate instances from a doc.

    labels : :obj:`list` of str
        Set of domain labels. If it is provided as a sequence, the
        order of labels is preserved ; otherwise its elements are
        sorted before storage. This guarantees a stable behaviour
        across runs and platforms, which greatly facilitates
        the comparability of models and results.

    zero : boolean, defaults to False
        If True, transform() will return the unknown label (UNK) for
        all (MM or only unrelated?) pairs.

    Attributes
    ----------
    labelset_ : dict(str, int)
        Map from labels to integers ; a few values are reserved:
        {UNK: 0, ROOT: 1, UNRELATED: 2}.
    """

    def __init__(self, instance_generator, labels, zero=False):
        self.instance_generator = instance_generator
        if not isinstance(labels, Sequence):
            labels = sorted(labels)
        self.labelset_ = {l: i for i, l in enumerate(labels, start=3)}
        self.labelset_[UNK] = 0
        self.labelset_[ROOT] = 1
        self.labelset_[UNRELATED] = 2
        self._zero = zero

    def transform(self, raw_documents):
        """Learn the label encoder and return a vector of labels

        There is one label per instance extracted from raw_documents.

        Parameters
        ----------
        raw_documents : list of ?
            Raw documents.

        Yields
        ------
        inst_lbls : list of int
            (Integer) label for each instance of the next document.
        """
        zlabel = UNK if self._zero else UNRELATED
        # run through documents to generate y
        for doc in raw_documents:
            inst_lbls = []
            for pair in self.instance_generator(doc):
                label = doc.relations.get(pair, zlabel)
                inst_lbls.append(self.labelset_[label])
            yield inst_lbls


# moved from educe.stac.learning.features
# FIXME refactor into a proper, consistent API:
# this code does a mix of responsibilities from DocumentPlus and other stuff

# ---------------------------------------------------------------------
# lexicon configuration
# ---------------------------------------------------------------------
class LexWrapper(object):
    """
    Configuration options for a given lexicon: where to find it,
    what to call it, what sorts of results to return
    """

    def __init__(self, key, filename, classes):
        """
        Note: classes=True means we want the (sub)-class of the lexical
        item found, and not just a general boolean
        """
        self.key = key
        self.filename = filename
        self.classes = classes
        self.lexicon = None

    def read(self, lexdir):
        """
        Read and store the lexicon as a mapping from words to their
        classes
        """
        path = os.path.join(lexdir, self.filename)
        self.lexicon = Lexicon.read_file(path)


LEXICONS = [
    LexWrapper('domain', 'stac_domain.txt', True),
    LexWrapper('robber', 'stac_domain2.txt', False),
    LexWrapper('trade', 'trade.txt', True),
    LexWrapper('dialog', 'dialog.txt', False),
    LexWrapper('opinion', 'opinion.txt', False),
    LexWrapper('modifier', 'modifiers.txt', False),
    # hand-extracted from trade prediction code, could
    # perhaps be merged with one of the other lexicons
    # fr.irit.stac.features.CalculsTraitsTache3
    LexWrapper('pronoun', 'pronouns.txt', True),
    LexWrapper('ref', 'stac_referential.txt', False)
]

# PDTB markers
PDTB_MARKERS_BASENAME = 'pdtb_markers.txt'

# VerbNet
VerbNetEntry = namedtuple("VerbNetEntry", "classname lemmas")

VERBNET_CLASSES = ['steal-10.5',
                   'get-13.5.1',
                   'give-13.1-1',
                   'want-32.1-1-1',
                   'want-32.1',
                   'exchange-13.6-1']

# Inquirer
INQUIRER_BASENAME = 'inqtabs.txt'

INQUIRER_CLASSES = ['Positiv',
                    'Negativ',
                    'Pstv',
                    'Ngtv',
                    'NegAff',
                    'PosAff',
                    'If',
                    'TrnGain',  # maybe gain/loss words related
                    'TrnLoss',  # ...transactions
                    'TrnLw',
                    'Food',    # related to Catan resources?
                    'Tool',    # related to Catan resources?
                    'Region',  # related to Catan game?
                    'Route']   # related to Catan game


# ---------------------------------------------------------------------
# preprocessing
# ---------------------------------------------------------------------
def player_addresees(edu):
    """
    The set of people spoken to during an edu annotation.
    This excludes known non-players, like 'All', or '?', or 'Please choose...',
    """
    addr1 = addressees(edu) or frozenset()
    return frozenset(x for x in addr1 if x not in ['All', '?'])


def players_for_doc(corpus, kdoc):
    """
    Return the set of speakers/addressees associated with a document.

    In STAC, documents are semi-arbitrarily cut into sub-documents for
    technical and possibly ergonomic reasons, ie. meaningless as far as we are
    concerned.  So to find all speakers, we would have to search all the
    subdocuments of a single document. ::

        (Corpus, String) -> Set String
    """
    speakers = set()
    docs = [corpus[k] for k in corpus if k.doc == kdoc]
    for doc in docs:
        for anno in doc.units:
            if educe.stac.is_turn(anno):
                turn_speaker = speaker(anno)
                if turn_speaker:
                    speakers.add(turn_speaker)
            elif educe.stac.is_edu(anno):
                speakers.update(player_addresees(anno))
    return frozenset(speakers)


# ---------------------------------------------------------------------
# feature extraction
# ---------------------------------------------------------------------

# The comments on these named tuples can be docstrings in Python3,
# or we can wrap the class, but eh...

# feature extraction environment
DocEnv = namedtuple("DocEnv", "inputs current sf_cache")

# Global resources and settings used to extract feature vectors
FeatureInput = namedtuple('FeatureInput',
                          ['corpus', 'postags', 'parses',
                           'lexicons', 'pdtb_lex',
                           'verbnet_entries',
                           'inquirer_lex'])

# A document and relevant contextual information
DocumentPlus = namedtuple('DocumentPlus',
                          ['key',
                           'doc',
                           'unitdoc',  # equiv doc from units
                           'players',
                           'parses'])


# ---------------------------------------------------------------------
# (single) feature cache
# ---------------------------------------------------------------------
class FeatureCache(dict):
    """
    Cache for single edu features.
    Retrieving an item from the cache lazily computes/memoises
    the single EDU features for it.
    """
    def __init__(self, inputs, current):
        self.inputs = inputs
        self.current = current
        super(FeatureCache, self).__init__()

    def __getitem__(self, edu):
        if edu.identifier() == ROOT:
            return KeyGroup('fake root group', [])
        elif edu in self:
            return super(FeatureCache, self).__getitem__(edu)
        else:
            vec = SingleEduKeys(self.inputs)
            vec.fill(self.current, edu)
            self[edu] = vec
            return vec

    def expire(self, edu):
        """
        Remove an edu from the cache if it's in there
        """
        if edu in self:
            del self[edu]


# ---------------------------------------------------------------------
# extraction generators
# ---------------------------------------------------------------------
def _get_unit_key(inputs, key):
    """
    Given the key for what is presumably a discourse level or
    unannotated document, return the key for its unit-level
    equivalent.
    """
    if key.annotator is None:
        twins = [k for k in inputs.corpus if
                 k.doc == key.doc and
                 k.subdoc == key.subdoc and
                 k.stage == 'units']
        return twins[0] if twins else None
    else:
        twin = copy.copy(key)
        twin.stage = 'units'
        return twin if twin in inputs.corpus else None


def mk_env(inputs, people, key):
    """Pre-process and bundle up a representation of the current doc.

    Parameters
    ----------
    inputs : FeatureInput
        Global information for feature extraction.

    people : dict(str, set(str))
        Set of people involved in the dialogue, for each game (map from
        document name to set of players).

    key : FileId
        Document identifier.

    Returns
    -------
    doc_env : DocEnv
        Representation of the designated document, ready for feature
        extraction.
    """
    doc = inputs.corpus[key]
    unit_key = _get_unit_key(inputs, key)
    current = DocumentPlus(key=key, doc=doc,
                           unitdoc=(inputs.corpus[unit_key] if unit_key
                                    else None),
                           players=people[key.doc],
                           parses=(inputs.parses[key] if inputs.parses
                                   else None))

    return DocEnv(inputs=inputs, current=current,
                  sf_cache=FeatureCache(inputs, current))


def get_players(inputs):
    """
    Return a dictionary mapping each document to the set of
    players in that document
    """
    kdocs = frozenset(k.doc for k in inputs.corpus)
    return {x: players_for_doc(inputs.corpus, x)
            for x in kdocs}


def relation_dict(doc, quiet=False):
    """
    Return the relations instances from a document in the
    form of an id pair to label dictionary

    If there is more than one relation between a pair of
    EDUs we pick one of them arbitrarily and ignore the
    other
    """
    relations = {}
    for rel in doc.relations:
        if not is_relation_instance(rel):
            # might be the odd Anaphora link lying around
            continue
        pair = rel.source.identifier(), rel.target.identifier()
        if pair not in relations:
            relations[pair] = rel.type
        elif not quiet:
            print(('Ignoring {type1} relation instance ({edu1} -> {edu2}); '
                   'another of type {type2} already exists'
                   '').format(type1=rel.type,
                              edu1=pair[0],
                              edu2=pair[1],
                              type2=relations[pair]),
                  file=sys.stderr)
    # generate fake root links
    for anno in doc.units:
        if not educe.stac.is_edu(anno):
            continue
        is_target = False
        for rel in doc.relations:
            if rel.target == anno:
                is_target = True
                break
        if not is_target:
            key = ROOT, anno.identifier()
            relations[key] = ROOT
    return relations


def _extract_pair(env, edu1, edu2):
    """Extract features for a given pair of EDUs.

    Directional, so would have to be called twice.
    """
    vec = PairKeys(env.inputs, sf_cache=env.sf_cache)
    vec.fill(env.current, edu1, edu2)
    return vec


def _mk_high_level_dialogues(current):
    """Helper to generate dialogues.

    Parameters
    ----------
    current : educe.stac.fusion.DocumentPlus
        Bundled representation of a document.

    Yields
    -------
    dia : `educe.stac.fusion.Dialogue`
        Next dialogue
    """
    doc = current.doc  # this is a GlozzDocument
    # first pass: create the EDU objects
    edus = sorted([x for x in doc.units if educe.stac.is_edu(x)],
                  key=lambda y: y.span)
    edus_in_dialogues = defaultdict(list)
    for edu in edus:
        edus_in_dialogues[edu.dialogue].append(edu)

    # finally, generate the high level dialogues
    relations = relation_dict(doc)
    dialogues = sorted(edus_in_dialogues, key=lambda x: x.span)
    for dia in dialogues:
        d_edus = edus_in_dialogues[dia]
        d_relations = {}
        for edu1, edu2 in itertools.product([FakeRootEDU] + d_edus, d_edus):
            id_pair = (edu1.identifier(), edu2.identifier())
            rel = relations.get(id_pair)
            if rel is not None:
                d_relations[(edu1, edu2)] = rel
        yield Dialogue(dia, d_edus, d_relations)


def mk_envs(inputs, stage):
    """Generate an environment for each document in the corpus
    within the given stage.

    The environment pools together all the information we
    have on a single document.

    Parameters
    ----------
    inputs : FeatureInput
        Global information used for feature extraction.

    stage : one of {'units', 'discourse', 'unannotated'}
        Annotation stage

    Yields
    -------
    env : DocEnv
        Next environment for feature extraction, one per doc.
    """
    people = get_players(inputs)
    for key in inputs.corpus:
        if key.stage != stage:
            continue
        yield mk_env(inputs, people, key)


def mk_high_level_dialogues(inputs, stage):
    """Generate all relevant EDU pairs for each designated document.

    Parameters
    ----------
    inputs : FeatureInput
        Named tuple of global resources and settings used to extract feature
        vectors.

    stage : string, one of {'units', 'discourse'}
        Stage of annotation

    Yields
    -------
    dia : `educe.stac.fusion.Dialogue`
        Next dialogue in the Dialogues
    """
    for env in mk_envs(inputs, stage):
        for dia in _mk_high_level_dialogues(env.current):
            yield dia


def extract_pair_features(inputs, stage):
    """Generator of feature vectors, one per pair of EDUs, in each dialogue.

    Parameters
    ----------
    inputs : FeatureInput
        Global parameters for feature extraction.

    stage : string, one of {'unannotated', 'units', 'discourse'}
        Annotation stage.

    Yields
    ------
    vecs : PairEduKeys
        List of feature vectors for the EDUs in the next document (here,
        STAC dialogue).
    """
    for env in mk_envs(inputs, stage):
        for dia in _mk_high_level_dialogues(env.current):
            vecs = []
            for edu1, edu2 in dia.edu_pairs():
                vec = _extract_pair(env, edu1, edu2)
                vecs.append(vec)
            yield vecs


# ---------------------------------------------------------------------
# extraction generators (single edu)
# ---------------------------------------------------------------------
def extract_single_features(inputs, stage, safety_check=True):
    """Generator of feature vectors, one per EDU, in each dialogue.

    Parameters
    ----------
    inputs : FeatureInput
        Global parameters for feature extraction.

    stage : string, one of {'unannotated', 'units', 'discourse'}
        Annotation stage.

    Yields
    ------
    vecs : SingleEduKeys
        List of feature vectors for the EDUs in the next document (in
        fact, STAC subdoc).
    """
    for env in mk_envs(inputs, stage):
        doc = env.current.doc
        if safety_check:
            # safety check: ensure all EDUs are processed
            # compare EDUs collected initially from the document, with
            # all EDUs processed during feature extraction
            all_edus = sorted([x for x in doc.units if educe.stac.is_edu(x)],
                              key=lambda y: y.span)
        processed_edus = []
        # skip any documents which are not yet annotated
        if env.current.unitdoc is None:
            continue
        # 2016-01-12 generate one list per Dialogue, rather than one per
        # subdoc
        for dia in _mk_high_level_dialogues(env.current):
            edus = dia.edus[1:]  # dia.edus[0]: left padding (fake root) EDU
            vecs = []
            for edu in edus:
                vec = SingleEduKeys(env.inputs)
                vec.fill(env.current, edu)
                vecs.append(vec)
                processed_edus.append(edu)  # safety check
            yield vecs
        if safety_check:
            # safety check: final step
            processed_edus = sorted(processed_edus, key=lambda y: y.span)
            assert processed_edus == all_edus


# ---------------------------------------------------------------------
# input readers
# ---------------------------------------------------------------------
def mk_is_interesting(args, single):
    """
    Return a function that filters corpus keys to pick out the ones
    we specified on the command line

    We have two cases here: for pair extraction, we just want to
    grab the units and if possible the discourse stage. In live mode,
    there won't be a discourse stage, but that's fine because we can
    just fall back on units.

    For single extraction (dialogue acts), we'll also want to grab the
    units stage and fall back to unannotated when in live mode. This
    is made a bit trickier by the fact that unannotated does not have
    an annotator, so we have to accomodate that.

    Phew.

    It's a bit specific to feature extraction in that here we are
    trying

    :type single: bool
    """
    if single:
        # ignore annotator filter for unannotated documents
        args1 = copy.copy(args)
        args1.annotator = None
        is_interesting1 = educe.util.mk_is_interesting(
            args1, preselected={'stage': ['unannotated']})
        # but pay attention to it for units
        args2 = args
        is_interesting2 = educe.util.mk_is_interesting(
            args2, preselected={'stage': ['units']})
        return lambda x: is_interesting1(x) or is_interesting2(x)
    else:
        preselected = {"stage": ["discourse", "units"]}
        return educe.util.mk_is_interesting(args, preselected=preselected)


def _fuse_corpus(corpus, postags):
    "Merge any dialogue/unit level documents together"
    to_delete = []
    for key in corpus:
        if key.stage == 'unannotated':
            # slightly abusive use of fuse_edus to just get the effect of
            # having EDUs that behave like contexts
            #
            # context: feature extraction for live mode dialogue acts
            # extraction, so by definition we don't have a units stage
            corpus[key] = fuse_edus(corpus[key], corpus[key], postags[key])
        elif key.stage == 'units':
            # similar Context-only abuse of fuse-edus (here, we have a units
            # stage but no dialogue to make use of)
            #
            # context: feature extraction for
            # - live mode discourse parsing (by definition we don't have a
            #   discourse stage yet, but we might have a units stage
            #   inferred earlier in the parsing pipeline)
            # - dialogue act annotation from corpus
            corpus[key] = fuse_edus(corpus[key], corpus[key], postags[key])
        elif key.stage == 'discourse':
            ukey = twin_key(key, 'units')
            corpus[key] = fuse_edus(corpus[key], corpus[ukey], postags[key])
            to_delete.append(ukey)
    for key in to_delete:
        del corpus[key]


def read_corpus_inputs(args):
    """Read and filter the part of the corpus we want features for.

    Parameters
    ----------
    args : Namespace
        Arguments given to the arg parser, in the form of a Namespace
        produced by `ArgumentParser.parse_args()`.

    Returns
    -------
    feat_input : FeatureInput
        Named tuple of global resources and settings used to extract feature
        vectors.
    """
    reader = educe.stac.Reader(args.corpus)
    anno_files = reader.filter(reader.files(),
                               mk_is_interesting(args, args.single))
    corpus = reader.slurp(anno_files, verbose=True)

    # optional: strip CDUs from the `GlozzDocument`s in the corpus
    if not args.ignore_cdus:
        # for all documents in the corpus, remove any CDUs and relink the
        # document according to the desired mode
        # this is performed on a graph model of the document:
        # `educe.stac.Graph.strip_cdus()` mutates the graph's doc
        for key in corpus:
            graph = stac_gr.Graph.from_doc(corpus, key)
            graph.strip_cdus(sloppy=True, mode=args.strip_mode)

    # read predicted POS tags, syntactic parse, coreferences etc.
    postags = postag.read_tags(corpus, args.corpus)
    parses = corenlp.read_results(corpus, args.corpus)
    _fuse_corpus(corpus, postags)

    # read our custom lexicons
    for lex in LEXICONS:
        lex.read(args.resources)
    # read lexicon PDTB discourse markers
    pdtb_lex_file = os.path.join(args.resources, PDTB_MARKERS_BASENAME)
    pdtb_lex = pdtb_markers.read_lexicon(pdtb_lex_file)
    # read Inquirer lexicon (disabled)
    # inq_txt_file = os.path.join(args.resources, INQUIRER_BASENAME)
    # inq_lex = inquirer.read_inquirer_lexicon(inq_txt_file, INQUIRER_CLASSES)
    inq_lex = {}

    verbnet_entries = [VerbNetEntry(x, frozenset(vnet.lemmas(x)))
                       for x in VERBNET_CLASSES]

    return FeatureInput(corpus=corpus,
                        postags=postags,
                        parses=parses,
                        lexicons=LEXICONS,
                        pdtb_lex=pdtb_lex,
                        verbnet_entries=verbnet_entries,
                        inquirer_lex=inq_lex)
