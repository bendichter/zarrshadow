"""Fetch files of the testing datasets on GIN with datalad.

The datasets are NEO's electrophysiology files and NeuroConv's optical
physiology files. Each is kept in one local folder, which an environment
variable can move. A file or folder that is already complete is used as it
is, without contacting GIN, which is sometimes slow or unreachable.

Run as a script, this downloads everything the tests marked gin read, which
continuous integration does once, ahead of the tests, on several runners:

    python tests/gin_data.py
"""

import functools
import os
import sys
import warnings

import pytest

DATASETS = {
    "ephys": ("https://gin.g-node.org/NeuralEnsemble/ephy_testing_data", "EPHY_TESTING_DATA", "~/ephy_testing_data"),
    "ophys": ("https://gin.g-node.org/CatalystNeuro/ophys_testing_data", "OPHYS_TESTING_DATA", "~/ophys_testing_data"),
}


def folder(dataset):
    """The local folder of a dataset: <NAME>_FOLDER, or its default in the home folder."""
    _, variable, default = DATASETS[dataset]
    return os.environ.get(f"{variable}_FOLDER") or os.path.expanduser(default)


def is_complete(path):
    """Whether a file or folder has its content. A file whose content was not downloaded is a broken link."""
    if os.path.isdir(path):
        return all(os.path.exists(os.path.join(root, name)) for root, _, names in os.walk(path) for name in names)
    return os.path.exists(path)


def _update_requested(dataset):
    """<NAME>_UPDATE=1 asks for an existing copy to be brought up to date."""
    return os.environ.get(f"{DATASETS[dataset][1]}_UPDATE", "") not in ("", "0")


@functools.cache
def _open(dataset):
    """The dataset as datalad sees it: installed if it is not there, and updated once if that was asked for."""
    datalad = pytest.importorskip("datalad.api")
    from datalad.support.gitrepo import GitRepo

    url, root = DATASETS[dataset][0], folder(dataset)
    if not (os.path.isdir(root) and GitRepo.is_valid_repo(root)):
        return datalad.install(path=root, source=url)
    opened = datalad.Dataset(root)
    if _update_requested(dataset):
        try:
            opened.update(merge=True)
        except Exception as e:  # GIN refuses or drops connections at times
            warnings.warn(f"Could not update {root} from GIN, so it is used as it is: {e}")
    return opened


@functools.cache
def fetch(dataset, relative_path):
    """Download a file or folder of a dataset if needed, and return its local path."""
    path = os.path.join(folder(dataset), relative_path)
    if _update_requested(dataset) or not is_complete(path):
        try:
            _open(dataset).get(relative_path, jobs=4)
        except Exception as e:
            if not is_complete(path):
                raise
            warnings.warn(f"Could not check {path} against GIN, so it is used as it is: {e}")
    return path


def required():
    """Every file or folder the tests marked gin read, as (dataset, relative path)."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import test_neo_gin
    import test_neuroconv_bridge

    return sorted(set(test_neo_gin.GIN_PATHS) | set(test_neuroconv_bridge.GIN_PATHS))


if __name__ == "__main__":
    for dataset, relative_path in required():
        print(f"{dataset}: {relative_path}", flush=True)
        fetch(dataset, relative_path)
