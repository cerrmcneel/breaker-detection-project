import cv2
import numpy as np
import pytest


def test_load_seeds_finds_images(tmp_path):
    """
    Given a folder with class subfolders containing images,
    SeedLibrary should index them by class name.
    """
    # 1. Create a fake 'MCB' subfolder with one dummy image
    mcb_dir = tmp_path / "MCB"
    mcb_dir.mkdir()
    cv2.imwrite(str(mcb_dir / "test.jpg"), np.zeros((10, 10, 3), dtype=np.uint8))

    # 2. Load seeds
    from src.data_gen.seed_library import SeedLibrary
    library = SeedLibrary()
    library.load_seeds(str(tmp_path))

    # 3. Assert the library indexed the class correctly
    assert 'MCB' in library.seeds
    assert len(library.seeds['MCB']) > 0

def test_get_random_seed_returns_ndarray(tmp_path):
    """
    get_random_seed() should return a numpy array (an image)
    for a known class.
    """
    # Setup
    mcb_dir = tmp_path / "MCB"
    mcb_dir.mkdir()
    cv2.imwrite(str(mcb_dir / "test.jpg"), np.zeros((50, 30, 3), dtype=np.uint8))

    from src.data_gen.seed_library import SeedLibrary
    library = SeedLibrary()
    library.load_seeds(str(tmp_path))

    # Act
    seed = library.get_random_seed('MCB')

    # Assert it is a valid image array
    assert isinstance(seed, np.ndarray)
    assert seed.ndim == 3  # Height x Width x Channels
