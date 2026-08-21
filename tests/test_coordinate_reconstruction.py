import numpy as np
import pytest

from evo2_distill.data.fasta import decode_tokens, encode_sequence, extract_window


def test_token_round_trip_and_unknown_base() -> None:
    assert decode_tokens(encode_sequence("ACGTNR")) == "ACGTNN"


def test_one_based_inclusive_window() -> None:
    assert extract_window("AACCGGTT", 2, 5) == "ACCG"
    with pytest.raises(ValueError):
        extract_window("ACGT", 2, 6)


def test_encoded_window_is_uint8() -> None:
    encoded = encode_sequence("ACGT" * 128)
    assert encoded.shape == (512,)
    assert encoded.dtype == np.uint8

