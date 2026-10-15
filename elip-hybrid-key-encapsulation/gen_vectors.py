"""Generate `vectors.json`, the HKE test vectors.

All inputs are derived from fixed labels, so the output is reproducible:

    .venv/bin/python gen_vectors.py
"""

import hashlib
import json
from pathlib import Path

from hke import (
    KEM,
    H_ecdh,
    HKEFailure,
    Decaps,
    EncapsDerand,
    G,
    cbytes,
    expandDecapsulationKey,
    lift,
    n,
)
from secp256k1lab.secp256k1 import FE  # importable once hke has set up the path

p = FE.SIZE

VECTORS = Path(__file__).with_name("vectors.json")


def label(s: str) -> bytes:
    return hashlib.sha256(b"ELIP-HKE/v1 test vector/" + s.encode()).digest()


def scalar(s: str) -> int:
    return int.from_bytes(label(s), "big") % (n - 1) + 1


def h(x: bytes) -> str:
    return x.hex()


def h32(x: int) -> str:
    return x.to_bytes(32, "big").hex()


def flip(data: bytes, i: int) -> bytes:
    return data[:i] + bytes([data[i] ^ 1]) + data[i + 1 :]


def set_coefficient(ek: bytes, i: int, value: int) -> bytes:
    """Overwrite the `i`-th 12-bit coefficient of `ek` (ByteEncode_12 layout)."""
    ek = bytearray(ek)
    j = 3 * (i // 2)
    if i % 2 == 0:
        ek[j] = value & 0xFF
        ek[j + 1] = (ek[j + 1] & 0xF0) | (value >> 8)
    else:
        ek[j + 1] = (ek[j + 1] & 0x0F) | ((value & 0x0F) << 4)
        ek[j + 2] = value >> 4
    return bytes(ek)


def valid_vector(comment: str, msk: bytes, e: int, m: bytes) -> dict:
    (b, dk, B, ek) = expandDecapsulationKey(msk)
    K, (E_bytes, c) = EncapsDerand(cbytes(B), ek, e, m)
    assert Decaps((E_bytes, c), msk) == K
    return {
        "comment": comment,
        "msk": h(msk),
        "e": h32(e),
        "m": h(m),
        "b": h32(b),
        "B": h(cbytes(B)),
        "ek": h(ek),
        "dk": h(dk),
        "E_bytes": h(E_bytes),
        "c": h(c),
        "K_c": h(H_ecdh(cbytes(b * lift(E_bytes)))),
        "K_q": h(KEM.Decaps(dk, c)),
        "K": h(K),
    }


def rejection_vector(comment: str, msk: bytes, E_bytes: bytes, c: bytes, K_honest: str) -> dict:
    K = Decaps((E_bytes, c), msk)
    assert h(K) != K_honest
    return {"comment": comment, "msk": h(msk), "E_bytes": h(E_bytes), "c": h(c), "K": h(K)}


def decaps_failure_vector(comment: str, msk: bytes, E_bytes: bytes, c: bytes) -> dict:
    try:
        Decaps((E_bytes, c), msk)
    except HKEFailure:
        return {"comment": comment, "msk": h(msk), "E_bytes": h(E_bytes), "c": h(c)}
    raise AssertionError(f"Decaps did not fail: {comment}")


def encaps_failure_vector(comment: str, B: bytes, ek: bytes, e: int, m: bytes) -> dict:
    try:
        EncapsDerand(B, ek, e, m)
    except HKEFailure:
        return {"comment": comment, "B": h(B), "ek": h(ek), "e": h32(e), "m": h(m)}
    raise AssertionError(f"EncapsDerand did not fail: {comment}")


def x_not_on_curve() -> bytes:
    X = 0
    while lift(b"\x02" + X.to_bytes(32, "big")) is not None:
        X += 1
    return b"\x02" + X.to_bytes(32, "big")


def x_ge_p() -> bytes:
    """A compressed encoding with X >= p whose X - p is a valid x coordinate,
    so an implementation that reduces X mod p instead of rejecting it accepts it."""
    X = p
    while lift(b"\x02" + (X - p).to_bytes(32, "big")) is None:
        X += 1
    return b"\x02" + X.to_bytes(32, "big")


def generate() -> dict:
    msk = [label(f"msk/{i}") for i in range(3)]
    valid = [
        valid_vector("generic", msk[0], scalar("e/0"), label("m/0")),
        valid_vector("generic, second recipient", msk[1], scalar("e/1"), label("m/1")),
        valid_vector("e = 1, so E = G", msk[2], 1, label("m/2")),
        valid_vector("e = n - 1, so E = -G", msk[2], n - 1, label("m/3")),
        valid_vector("m = 0^32", msk[0], scalar("e/4"), bytes(32)),
    ]

    v0, v1 = valid[0], valid[1]
    E0, c0, K0 = bytes.fromhex(v0["E_bytes"]), bytes.fromhex(v0["c"]), v0["K"]
    E1, c1 = bytes.fromhex(v1["E_bytes"]), bytes.fromhex(v1["c"])
    implicit_rejection = [
        rejection_vector("valid vector 0, first bit of c flipped", msk[0], E0, flip(c0, 0), K0),
        rejection_vector("valid vector 0, last byte of c flipped", msk[0], E0, flip(c0, KEM.L_c - 1), K0),
        rejection_vector("valid vector 0 with E replaced by 3*G", msk[0], cbytes(3 * G), c0, K0),
        rejection_vector("valid vector 0 with E from valid vector 1", msk[0], E1, c0, K0),
        rejection_vector("valid vector 1 decapsulated by recipient 0", msk[0], E1, c1, K0),
    ]

    decaps_failure = [
        decaps_failure_vector("E_bytes empty", msk[0], b"", c0),
        decaps_failure_vector("E_bytes = 0x00 (point at infinity)", msk[0], b"\x00", c0),
        decaps_failure_vector("E_bytes prefix 0x04", msk[0], b"\x04" + E0[1:], c0),
        decaps_failure_vector("E_bytes 32 bytes", msk[0], E0[:32], c0),
        decaps_failure_vector("E_bytes 34 bytes", msk[0], E0 + b"\x00", c0),
        decaps_failure_vector("E_bytes X = p + 1 (X - p is a valid x)", msk[0], x_ge_p(), c0),
        decaps_failure_vector("E_bytes X not on the curve", msk[0], x_not_on_curve(), c0),
        decaps_failure_vector("|c| = L_c - 1", msk[0], E0, c0[:-1]),
        decaps_failure_vector("|c| = L_c + 1", msk[0], E0, c0 + b"\x00"),
    ]

    B0, ek0 = bytes.fromhex(v0["B"]), bytes.fromhex(v0["ek"])
    e0, m0 = int(v0["e"], 16), bytes.fromhex(v0["m"])
    encaps_failure = [
        encaps_failure_vector("ek coefficient 0 = q", B0, set_coefficient(ek0, 0, KEM.q), e0, m0),
        encaps_failure_vector("ek coefficient 1 = 4095", B0, set_coefficient(ek0, 1, 4095), e0, m0),
        encaps_failure_vector("|ek| = L_ek - 1", B0, ek0[:-1], e0, m0),
        encaps_failure_vector("|ek| = L_ek + 1", B0, ek0 + b"\x00", e0, m0),
        encaps_failure_vector("B empty", b"", ek0, e0, m0),
        encaps_failure_vector("B = 0x00 (point at infinity)", b"\x00", ek0, e0, m0),
        encaps_failure_vector("B prefix 0x04", b"\x04" + B0[1:], ek0, e0, m0),
        encaps_failure_vector("B uncompressed", lift(B0).to_bytes_uncompressed(), ek0, e0, m0),
        encaps_failure_vector("B 32 bytes", B0[:32], ek0, e0, m0),
        encaps_failure_vector("B 34 bytes", B0 + b"\x00", ek0, e0, m0),
        encaps_failure_vector("B X = p + 1 (X - p is a valid x)", x_ge_p(), ek0, e0, m0),
        encaps_failure_vector("B X not on the curve", x_not_on_curve(), ek0, e0, m0),
    ]

    return {
        "scheme": "ELIP-HKE/v1, ML-KEM-768",
        "encoding": "byte strings and scalars are lowercase hex; scalars are 32-byte big-endian",
        "valid": valid,
        "implicit_rejection": implicit_rejection,
        "decaps_failure": decaps_failure,
        "encaps_failure": encaps_failure,
    }


if __name__ == "__main__":
    VECTORS.write_text(json.dumps(generate(), indent=2) + "\n")
    print(f"wrote {VECTORS}")
