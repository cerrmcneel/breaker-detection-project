"""Pixel-content fingerprinting shared by the gateway and maintenance tools.

Kept in its own dependency-light module (OpenCV + NumPy + stdlib only) so CLI
tools can compute the exact same fingerprint as ``app/main.py`` without
importing FastAPI, loading ``.env``, or mounting static directories.
"""

import hashlib
from typing import Optional

import cv2
import numpy as np


def image_content_hash(file_bytes: bytes) -> Optional[str]:
    """Fingerprint an image by its DECODED PIXELS, not its file bytes.

    Hashing the raw file makes de-duplication trivially defeatable by anything
    that rewrites metadata. Measured 2026-08-11 on a real case: a photo already
    in raw_uploads was re-exported with a different EXIF header -- same
    4032x2268 pixels, same 1,780,560 byte length, byte-identical compressed
    scan data, and only the first 64 KB differed. The file hash changed, so the
    same panel would have been stored twice.

    Decoded at 1/4 scale on purpose. Metadata never affects pixels, so the
    reduced image still collapses the metadata-rewrite case exactly, while
    being ~4x cheaper than a full decode (20 ms vs 80 ms per photo here). That
    matters because get_unique_dataset_count() fingerprints the whole dataset,
    and the cost grows with it.

    The image shape is folded in so two differently-sized images can never
    collide on pixel bytes alone. Returns None if the bytes will not decode.
    """
    img = cv2.imdecode(np.frombuffer(file_bytes, dtype=np.uint8), cv2.IMREAD_REDUCED_COLOR_4)
    if img is None:
        return None
    arr = np.ascontiguousarray(img)
    digest = hashlib.sha256()
    digest.update(str(arr.shape).encode())
    digest.update(arr.tobytes())
    return digest.hexdigest()
