"""Guards against train/val contamination in the real dataset.

An image present in both splits makes validation partly a memorisation test, and
the resulting number looks fine while measuring the wrong thing -- the same
failure mode as the classes.txt label permutation, which scored 92% against
broken ground truth.

`data/` is gitignored, so these tests SKIP in CI and on a fresh clone and only do
real work on a machine that actually holds the dataset (i.e. before training).
The split-tool tests below run everywhere, since they build their own fixtures.
"""
import hashlib
import os
import shutil

import pytest

from src.tools.split_validation import file_hash, hash_dir, split_validation_set

TRAIN_IMG_DIR = os.path.join("data", "dataset", "train", "images")
VAL_IMG_DIR = os.path.join("data", "dataset", "val", "images")


def _require_dataset():
    if not (os.path.isdir(TRAIN_IMG_DIR) and os.path.isdir(VAL_IMG_DIR)):
        pytest.skip("dataset not present (data/ is gitignored) -- nothing to check")


# --- the real dataset, when it is present -------------------------------------------

def test_no_real_image_appears_in_both_train_and_val():
    _require_dataset()
    train = hash_dir(TRAIN_IMG_DIR)
    val = hash_dir(VAL_IMG_DIR)
    overlap = set(train) & set(val)

    if overlap:
        detail = "\n".join(
            f"  {train[h][0]}  (train)  ==  {val[h][0]}  (val)" for h in sorted(overlap)[:10]
        )
        pytest.fail(
            f"{len(overlap)} image(s) are in BOTH train and val "
            f"({100 * len(overlap) / max(1, len(val)):.0f}% of val is contaminated).\n"
            f"Validation is partly measuring memorisation.\n{detail}"
        )


def test_synthetic_images_do_not_cross_splits():
    """Synthetic panels are generated, so an overlap means a bad copy, not chance."""
    _require_dataset()
    train = hash_dir(TRAIN_IMG_DIR, skip_synthetic=False)
    val = hash_dir(VAL_IMG_DIR, skip_synthetic=False)
    assert not (set(train) & set(val)), "identical images across splits (including synthetic)"


def test_validation_split_is_not_empty():
    """A guard against a rebuild that silently drains val."""
    _require_dataset()
    assert hash_dir(VAL_IMG_DIR), "validation split has no real images"


# --- the split tool itself (runs everywhere, builds its own data) ---------------------

def _png(tmp, path, colour):
    import cv2
    import numpy as np
    img = np.full((8, 8, 3), colour, dtype=np.uint8)
    full = os.path.join(tmp, path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    cv2.imwrite(full, img)
    return full


@pytest.fixture
def dataset(tmp_path, monkeypatch):
    """A miniature data/ tree, with the tool pointed at it."""
    root = str(tmp_path)
    monkeypatch.chdir(root)
    for d in ("data/images/raw_uploads", "data/dataset/train/images",
              "data/dataset/val/images", "data/dataset/val/labels"):
        os.makedirs(os.path.join(root, d), exist_ok=True)
    return root


def test_split_moves_rather_than_copies(dataset):
    """The original bug: copying left the image in raw_uploads to be mixed into train."""
    src = _png(dataset, "data/images/raw_uploads/a.png", 10)
    open(os.path.join(dataset, "data/images/raw_uploads/a.txt"), "w").close()

    moved, _ = split_validation_set()

    assert len(moved) == 1
    assert not os.path.exists(src), "source still in raw_uploads -- it was copied, not moved"
    assert os.path.exists(os.path.join(dataset, "data/dataset/val/images/a.png"))
    assert os.path.exists(os.path.join(dataset, "data/dataset/val/labels/a.txt"))


def test_split_skips_an_image_already_in_train(dataset):
    """Same content under a different name must not be pulled into val."""
    _png(dataset, "data/dataset/train/images/already.png", 42)
    shutil.copy(os.path.join(dataset, "data/dataset/train/images/already.png"),
                os.path.join(dataset, "data/images/raw_uploads/renamed_copy.png"))

    moved, skipped = split_validation_set()

    assert moved == []
    assert [why for _, why in skipped] == ["already in train"]
    assert not os.path.exists(os.path.join(dataset, "data/dataset/val/images/renamed_copy.png"))


def test_split_does_not_move_two_copies_of_one_image_in_the_same_run(dataset):
    """Byte-identical files arriving under two names must yield ONE val entry."""
    _png(dataset, "data/images/raw_uploads/one.png", 7)
    shutil.copy(os.path.join(dataset, "data/images/raw_uploads/one.png"),
                os.path.join(dataset, "data/images/raw_uploads/two.png"))

    moved, skipped = split_validation_set()

    assert len(moved) == 1, "the duplicate should have been skipped within the same run"
    assert [why for _, why in skipped] == ["already in val"]


def test_split_result_is_hash_disjoint(dataset):
    """The property the whole tool exists to guarantee."""
    _png(dataset, "data/dataset/train/images/t1.png", 1)
    for i, colour in enumerate((2, 3, 4)):
        _png(dataset, f"data/images/raw_uploads/r{i}.png", colour)
    shutil.copy(os.path.join(dataset, "data/dataset/train/images/t1.png"),
                os.path.join(dataset, "data/images/raw_uploads/sneaky.png"))

    split_validation_set()

    train = set(hash_dir(os.path.join(dataset, "data/dataset/train/images")))
    val = set(hash_dir(os.path.join(dataset, "data/dataset/val/images")))
    assert train & val == set()
    assert len(val) == 3, "the three genuinely new images should be in val"


def test_dry_run_changes_nothing(dataset):
    src = _png(dataset, "data/images/raw_uploads/a.png", 10)
    moved, _ = split_validation_set(dry_run=True)

    assert len(moved) == 1
    assert os.path.exists(src), "dry run moved a file"
    assert not os.listdir(os.path.join(dataset, "data/dataset/val/images"))


def test_file_hash_matches_hashlib(dataset):
    """The chunked reader must agree with a plain one-shot hash."""
    p = _png(dataset, "data/images/raw_uploads/a.png", 99)
    assert file_hash(p) == hashlib.sha256(open(p, "rb").read()).hexdigest()


def test_the_contamination_guard_actually_detects_contamination(dataset):
    """Proves the real-dataset test above is not vacuous.

    Recreates the exact shape of the VM's dataset -- the same image in train and
    val -- and asserts the detection the guard relies on fires.
    """
    _png(dataset, "data/dataset/train/images/panel.png", 5)
    shutil.copy(os.path.join(dataset, "data/dataset/train/images/panel.png"),
                os.path.join(dataset, "data/dataset/val/images/panel.png"))
    # ...and the same picture again under a different name, as really happened.
    shutil.copy(os.path.join(dataset, "data/dataset/train/images/panel.png"),
                os.path.join(dataset, "data/dataset/val/images/renamed.png"))

    train = hash_dir(os.path.join(dataset, "data/dataset/train/images"))
    val = hash_dir(os.path.join(dataset, "data/dataset/val/images"))
    overlap = set(train) & set(val)

    assert overlap, "guard failed to notice an image present in both splits"
    assert len(val) == 1, "both val filenames hash to the same content"
