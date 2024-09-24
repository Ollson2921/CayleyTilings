"""Tests for the CayleyPermutation class."""

import pytest
from cayley_permutations import CayleyPermutation


def test_init_method():
    """Tests the init method."""
    cperm1 = CayleyPermutation([0, 1, 2, 3])
    assert cperm1 == CayleyPermutation((0, 1, 2, 3))
    assert cperm1.cperm == (0, 1, 2, 3)

    with pytest.raises(TypeError):
        CayleyPermutation(cperm1)
