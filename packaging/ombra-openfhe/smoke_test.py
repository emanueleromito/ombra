"""Smoke test of an installed ombra-openfhe wheel: a CKKS encrypt, multiply, decrypt round trip.

Synthetic data only. Run by cibuildwheel as the test-command of this package.
"""

import openfhe as ofhe

assert ofhe.__version__ == "1.6.1.0", ofhe.__version__
params = ofhe.CCParamsCKKSRNS()
params.SetMultiplicativeDepth(2)
params.SetScalingModSize(50)
params.SetBatchSize(8)
params.SetSecurityLevel(ofhe.SecurityLevel.HEStd_128_classic)
cc = ofhe.GenCryptoContext(params)
cc.Enable(ofhe.PKESchemeFeature.PKE)
cc.Enable(ofhe.PKESchemeFeature.LEVELEDSHE)
keys = cc.KeyGen()
cc.EvalMultKeyGen(keys.secretKey)
x = [0.5, 1.0, -2.0, 3.0]
ct = cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(x))
sq = cc.EvalMult(ct, ct)
out = cc.Decrypt(sq, keys.secretKey)
out.SetLength(len(x))
got = out.GetRealPackedValue()
assert all(abs(g - v * v) < 1e-3 for g, v in zip(got, x, strict=True)), got
print("ombra-openfhe smoke test OK", ofhe.__version__)
