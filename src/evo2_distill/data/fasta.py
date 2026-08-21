from __future__ import annotations

from pathlib import Path

import numpy as np

VOCAB = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}
INVERSE_VOCAB = np.asarray(["A", "C", "G", "T", "N"])
MASK_TOKEN = 5


def encode_sequence(sequence: str) -> np.ndarray:
    """Encode A/C/G/T/N to the frozen uint8 vocabulary; other bases become N."""
    return np.fromiter(
        (VOCAB.get(base, VOCAB["N"]) for base in sequence.upper()),
        dtype=np.uint8,
    )


def decode_tokens(tokens: np.ndarray) -> str:
    values = np.asarray(tokens, dtype=np.int64)
    if np.any((values < 0) | (values > 4)):
        raise ValueError("Only A/C/G/T/N tokens can be decoded")
    return "".join(INVERSE_VOCAB[values].tolist())


def load_fasta(path: str | Path) -> dict[str, str]:
    records: dict[str, list[str]] = {}
    current: str | None = None
    with Path(path).open("r", encoding="ascii") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                current = line[1:].split()[0]
                if current in records:
                    raise ValueError(f"Duplicate FASTA record {current} in {path}")
                records[current] = []
            elif current is None:
                raise ValueError(f"Sequence before FASTA header in {path}")
            else:
                records[current].append(line.upper())
    return {record_id: "".join(parts) for record_id, parts in records.items()}


def extract_window(sequence: str, start: int, end: int) -> str:
    """Extract a frozen 1-based inclusive interval."""
    if start < 1 or end < start:
        raise ValueError(f"Invalid 1-based inclusive interval: {start}-{end}")
    window = sequence[start - 1 : end]
    expected = end - start + 1
    if len(window) != expected:
        raise ValueError(
            f"Interval {start}-{end} exceeds sequence bounds; expected {expected}, found {len(window)}"
        )
    return window

