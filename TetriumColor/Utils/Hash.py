import hashlib


def stable_hash(obj) -> int:
    """Returns a stable SHA-256 hash of a string representation of the input object."""
    obj_str = str(obj).encode('utf-8')
    return int(hashlib.sha256(obj_str).hexdigest(), 16)
