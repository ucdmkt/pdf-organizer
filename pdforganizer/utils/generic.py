"""Generic helper utility functions."""

import hashlib


def compute_file_hash(path):
    """
    Computes the SHA-256 hash of a file's content.
    Reads the file in binary mode to ensure content-only hashing.
    """
    sha256_hash = hashlib.sha256()
    with open(path, "rb") as f:
        # Read and update hash string value in blocks of 4K
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()
