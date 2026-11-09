import json
import math

import pytest

pytestmark = pytest.mark.openfhe


def test_probe(tmp_path):
    import openfhe as ofhe

    print("ScalingTechnique:", [a for a in dir(ofhe.ScalingTechnique) if a.isupper()])
    print(
        "params attrs:",
        [a for a in dir(ofhe.CCParamsCKKSRNS) if "omposite" in a or "egister" in a],
    )
    print("cc attrs:", [a for a in dir(ofhe.CryptoContext) if "omposite" in a or "Scal" in a])
    for first in (62, 64):
        p = ofhe.CCParamsCKKSRNS()
        p.SetSecurityLevel(ofhe.SecurityLevel.HEStd_128_classic)
        p.SetRingDim(16384)
        p.SetMultiplicativeDepth(2)
        p.SetScalingModSize(60)
        p.SetFirstModSize(first)
        p.SetBatchSize(8192)
        p.SetScalingTechnique(ofhe.ScalingTechnique.COMPOSITESCALINGMANUAL)
        p.SetKeySwitchTechnique(ofhe.KeySwitchTechnique.HYBRID)
        p.SetCompositeDegree(2)
        p.SetRegisterWordSize(32)
        cc = ofhe.GenCryptoContext(p)
        for f in (
            ofhe.PKESchemeFeature.PKE,
            ofhe.PKESchemeFeature.KEYSWITCH,
            ofhe.PKESchemeFeature.LEVELEDSHE,
        ):
            cc.Enable(f)
        keys = cc.KeyGen()
        cc.EvalMultKeyGen(keys.secretKey)
        kf = tmp_path / "k.json"
        ofhe.CryptoContext.SerializeEvalMultKey(str(kf), ofhe.JSON, keys.publicKey.GetKeyTag())
        tree = json.loads(kf.read_text())
        entry = tree["value0"][0]["value"][0]["ptr_wrapper"]["data"]
        qp = [int(t["v"]["ptr_wrapper"]["data"]["m"]["v"]) for t in entry["ak"][0]["v"]]
        print(
            "first",
            first,
            "towers",
            len(qp),
            "bits",
            [round(math.log2(m), 2) for m in qp],
            "digits",
            len(entry["ak"]),
            "logQP",
            sum(math.log2(m) for m in qp),
            "N",
            cc.GetRingDimension(),
        )
