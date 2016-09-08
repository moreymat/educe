"""This module enables to load the output of the discourse parser described in
(Feng & Hirst 2014).

"""

import codecs
import glob
import os

from .parse import parse_rst_dt_tree


def load_feng_output_files(root_dir):
    """Load ctrees output by Feng & Hirst's parser on the TEST section of
    RST-WSJ.

    Parameters
    ----------
    root_dir: string
        Path to the main folder containing the parser's output

    Returns
    -------
    data: dict
        Dictionary that should be akin to a sklearn Bunch, with
        interesting keys 'filenames', 'doc_names' and 'rst_ctrees'.

    Notes
    -----
    To ensure compatibility with the rest of the code base, doc_names
    are automatically added the ".out" extension. This would not work
    for fileX documents, but they are absent from the TEST section of
    the RST-WSJ treebank.
    """
    # find all files with the right extension
    file_ext = '.txt.dis'
    pathname = os.path.join(root_dir, '*{}'.format(file_ext))
    # filenames are sorted by name to avoid having to realign data
    # loaded with different functions
    filenames = sorted(glob.glob(pathname))  # glob.glob() returns a list

    # find corresponding doc names
    doc_names = [os.path.basename(filename).rsplit('.', 2)[0] + '.out'
                 for filename in filenames]

    # load the RST trees
    rst_ctrees = []
    for filename in filenames:
        with codecs.open(filename, 'r', 'utf-8') as f:
            # TODO (?) add support for and use RSTContext
            rst_ctree = parse_rst_dt_tree(f.read(), None)
            rst_ctrees.append(rst_ctree)

    data = dict(filenames=filenames,
                doc_names=doc_names,
                rst_ctrees=rst_ctrees)

    return data
