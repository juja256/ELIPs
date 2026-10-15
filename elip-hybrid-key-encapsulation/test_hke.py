import hashlib

import pytest

from kyber_py.ml_kem import ML_KEM_768 as kyber

from gen_vectors import set_coefficient
from hke import (
    H,
    KEM,
    L_c,
    L_ek,
    HKEFailure,
    Decaps,
    Encaps,
    G,
    KeyGen,
    cbytes,
    expandDecapsulationKey,
    lift,
    n,
)
from secp256k1lab.secp256k1 import FE  # importable once hke has set up the path

p = FE.SIZE


@pytest.fixture(scope="module")
def recipient():
    msk, (B, ek) = KeyGen()
    return msk, B, ek


@pytest.fixture(scope="module")
def encapsulation(recipient):
    _, B, ek = recipient
    return Encaps(B, ek)


def flip(data: bytes, i: int) -> bytes:
    return data[:i] + bytes([data[i] ^ 1]) + data[i + 1 :]


def _x_not_on_curve() -> bytes:
    X = 0
    while lift(b"\x02" + X.to_bytes(32, "big")) is not None:
        X += 1
    return b"\x02" + X.to_bytes(32, "big")


# --- secp256k1 and tagged hash ----------------------------------------------


def test_generator_encoding():
    assert cbytes(G).hex() == (
        "0279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798"
    )
    assert cbytes(2 * G).hex() == (
        "02c6047f9441ed7d6d3045406e95c07cd85c778e4b8cef3ca7abac09b95c709ee5"
    )


def test_lift_inverts_cbytes():
    for k in (1, 2, 3, 0xDEADBEEF, n - 1):
        P = k * G
        assert lift(cbytes(P)) == P


def test_lift_rejects_noncanonical_x():
    # X = p + 1 reduces to x = 1, a valid x coordinate; lift must not reduce.
    assert lift(b"\x02" + (1).to_bytes(32, "big")) is not None
    assert lift(b"\x02" + (p + 1).to_bytes(32, "big")) is None


def test_lift_rejects_uncompressed_and_hybrid_encodings():
    # lift accepts only the 33-byte compressed form.
    uncompressed = G.to_bytes_uncompressed()
    assert len(uncompressed) == 65
    for prefix in (4, 6, 7):
        assert lift(bytes([prefix]) + uncompressed[1:]) is None


@pytest.mark.parametrize(
    "tag",
    [
        "ELIP-HKE/v1/blind",
        "ELIP-HKE/v1/kem-d",
        "ELIP-HKE/v1/kem-z",
        "ELIP-HKE/v1/ecdh",
        "ELIP-HKE/v1/seed/ML-KEM-768",
    ],
)
def test_tagged_hash_matches_bip340_definition(tag):
    msg = bytes(range(100))
    t = hashlib.sha256(tag.encode()).digest()
    assert H(tag, msg) == hashlib.sha256(t + t + msg).digest()


# --- Correctness ------------------------------------------------------------


def test_keygen_is_deterministic_from_msk(recipient):
    msk, B, ek = recipient
    (_, dk1, B1, ek1) = expandDecapsulationKey(msk)
    (_, dk2, B2, ek2) = expandDecapsulationKey(msk)
    assert (B1, ek1, dk1) == (B2, ek2, dk2)
    assert (cbytes(B1), ek1) == (B, ek)


def test_sizes(recipient, encapsulation):
    _, _, ek = recipient
    K, (E_bytes, c) = encapsulation
    assert len(ek) == L_ek
    assert len(c) == L_c
    assert len(E_bytes) == 33
    assert lift(E_bytes) is not None
    assert len(K) == 32


def test_roundtrip(recipient, encapsulation):
    msk, _, _ = recipient
    K, ct = encapsulation
    assert Decaps(ct, msk) == K


@pytest.mark.parametrize("_", range(3))
def test_roundtrip_fresh_keys(_):
    msk, (B, ek) = KeyGen()
    K, ct = Encaps(B, ek)
    assert Decaps(ct, msk) == K


def test_encaps_is_randomized(recipient):
    _, B, ek = recipient
    K1, (E1, c1) = Encaps(B, ek)
    K2, (E2, c2) = Encaps(B, ek)
    assert K1 != K2 and E1 != E2 and c1 != c2


@pytest.mark.parametrize(
    "B_bytes",
    [
        b"",
        b"\x00",  # SEC1 encoding of O
        b"\x04" + bytes(32),  # bad prefix
        G.to_bytes_uncompressed(),
        b"\x02" + bytes(31),  # too short
        b"\x02" + bytes(33),  # too long
        b"\x02" + (p + 1).to_bytes(32, "big"),  # X >= p, X - p a valid x
        _x_not_on_curve(),
    ],
    ids=["empty", "infinity", "bad-prefix", "uncompressed", "short", "long", "x-ge-p", "off-curve"],
)
def test_encaps_rejects_invalid_B(recipient, B_bytes):
    _, _, ek = recipient
    with pytest.raises(HKEFailure):
        Encaps(B_bytes, ek)


# --- Encapsulation key check (FIPS 203, 7.2) --------------------------------


def kyber_accepts(ek: bytes) -> bool:
    try:
        kyber.encaps(ek)
        return True
    except ValueError:
        return False


def test_check_ek_accepts_valid_key(recipient):
    _, _, ek = recipient
    assert KEM.CheckEK(ek)


@pytest.mark.parametrize("i", [0, 1, 767])
@pytest.mark.parametrize("value", [KEM.q - 1, KEM.q, 4095])
def test_check_ek_modulus_check(recipient, i, value):
    _, _, ek = recipient
    bad = set_coefficient(ek, i, value)
    assert KEM.CheckEK(bad) == (value < KEM.q)
    assert KEM.CheckEK(bad) == kyber_accepts(bad)


@pytest.mark.parametrize("length", [0, L_ek - 1, L_ek + 1])
def test_encaps_rejects_wrong_length_ek(recipient, length):
    _, B, ek = recipient
    with pytest.raises(HKEFailure):
        Encaps(B, (ek * 2)[:length])


def test_encaps_rejects_ek_failing_modulus_check(recipient):
    _, B, ek = recipient
    with pytest.raises(HKEFailure):
        Encaps(B, set_coefficient(ek, 5, KEM.q))


# --- Invalid ciphertexts: implicit rejection (wrong K, no error) -------------


@pytest.mark.parametrize("i", [0, 1, L_c // 2, L_c - 1])
def test_tampered_c_gives_wrong_key(recipient, encapsulation, i):
    msk, _, _ = recipient
    K, (E_bytes, c) = encapsulation
    K_bad = Decaps((E_bytes, flip(c, i)), msk)
    assert K_bad != K
    assert len(K_bad) == 32


def test_implicit_rejection_is_deterministic(recipient, encapsulation):
    msk, _, _ = recipient
    _, (E_bytes, c) = encapsulation
    bad = (E_bytes, flip(c, 7))
    assert Decaps(bad, msk) == Decaps(bad, msk)


def test_different_valid_E_gives_wrong_key(recipient, encapsulation):
    msk, _, _ = recipient
    K, (_, c) = encapsulation
    other_E = cbytes(12345 * G)
    assert Decaps((other_E, c), msk) != K


def test_ciphertext_for_other_recipient_gives_wrong_key(encapsulation):
    K, ct = encapsulation
    other_msk, _ = KeyGen()
    assert Decaps(ct, other_msk) != K


def test_ciphertexts_cannot_be_mixed(recipient):
    msk, B, ek = recipient
    K1, (E1, c1) = Encaps(B, ek)
    K2, (E2, c2) = Encaps(B, ek)
    K_mixed = Decaps((E1, c2), msk)
    assert K_mixed not in (K1, K2)


# --- Invalid ciphertexts: explicit failure (format checks) -------------------


@pytest.mark.parametrize("length", [0, L_c - 1, L_c + 1])
def test_wrong_length_c_fails(recipient, encapsulation, length):
    msk, _, _ = recipient
    _, (E_bytes, c) = encapsulation
    bad_c = (c * 2)[:length]
    with pytest.raises(HKEFailure):
        Decaps((E_bytes, bad_c), msk)


@pytest.mark.parametrize(
    "E_bytes",
    [
        b"",
        b"\x00",  # SEC1 encoding of O
        b"\x04" + bytes(32),  # bad prefix
        b"\x02" + bytes(31),  # too short
        b"\x02" + bytes(33),  # too long
        b"\x02" + (p + 1).to_bytes(32, "big"),  # X >= p, X - p a valid x
        _x_not_on_curve(),
    ],
    ids=["empty", "infinity", "bad-prefix", "short", "long", "x-ge-p", "off-curve"],
)
def test_invalid_E_fails(recipient, encapsulation, E_bytes):
    msk, _, _ = recipient
    _, (_, c) = encapsulation
    with pytest.raises(HKEFailure):
        Decaps((E_bytes, c), msk)
