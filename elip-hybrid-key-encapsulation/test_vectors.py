import json

import pytest

from gen_vectors import VECTORS, generate
from hke import (
    KEM,
    H_ecdh,
    HKEFailure,
    Decaps,
    EncapsDerand,
    cbytes,
    expandDecapsulationKey,
    lift,
)

V = json.loads(VECTORS.read_text())


def x(s: str) -> bytes:
    return bytes.fromhex(s)


def ids(group):
    return [v["comment"] for v in V[group]]


def test_vectors_file_is_current():
    assert generate() == V


@pytest.mark.parametrize("v", V["valid"], ids=ids("valid"))
def test_valid(v):
    msk = x(v["msk"])

    (b, dk, B, ek) = expandDecapsulationKey(msk)
    assert b == int(v["b"], 16)
    assert cbytes(B) == x(v["B"])
    assert ek == x(v["ek"])
    assert dk == x(v["dk"])

    K, (E_bytes, c) = EncapsDerand(x(v["B"]), x(v["ek"]), int(v["e"], 16), x(v["m"]))
    assert E_bytes == x(v["E_bytes"])
    assert c == x(v["c"])
    assert K == x(v["K"])

    assert H_ecdh(cbytes(b * lift(E_bytes))) == x(v["K_c"])
    assert KEM.Decaps(dk, c) == x(v["K_q"])

    assert Decaps((x(v["E_bytes"]), x(v["c"])), msk) == x(v["K"])


@pytest.mark.parametrize("v", V["implicit_rejection"], ids=ids("implicit_rejection"))
def test_implicit_rejection(v):
    assert Decaps((x(v["E_bytes"]), x(v["c"])), x(v["msk"])) == x(v["K"])


@pytest.mark.parametrize("v", V["decaps_failure"], ids=ids("decaps_failure"))
def test_decaps_failure(v):
    with pytest.raises(HKEFailure):
        Decaps((x(v["E_bytes"]), x(v["c"])), x(v["msk"]))


@pytest.mark.parametrize("v", V["encaps_failure"], ids=ids("encaps_failure"))
def test_encaps_failure(v):
    with pytest.raises(HKEFailure):
        EncapsDerand(x(v["B"]), x(v["ek"]), int(v["e"], 16), x(v["m"]))
