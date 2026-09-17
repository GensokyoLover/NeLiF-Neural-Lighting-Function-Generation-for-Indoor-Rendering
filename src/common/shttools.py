"""Compressed scene and light data loading for NeLiF."""

import pickle

import zstandard as zstd


def load_pklzst(filename):
    """Load a Zstandard-compressed pickle, reporting the path on failure."""
    try:
        with open(filename, "rb") as f:
            compressed = f.read()

        dctx = zstd.ZstdDecompressor()
        raw = dctx.decompress(compressed)

        data = pickle.loads(raw)
        return data

    except Exception as e:
        print("=" * 80)
        print("[ERROR] Failed to load pkl.zst file:")
        print(filename)
        print("Error type:", type(e).__name__)
        print("Error msg :", str(e))
        print("=" * 80)

        raise RuntimeError(
            f"Failed to load pkl.zst file: {filename} | "
            f"{type(e).__name__}: {str(e)}"
        ) from e
