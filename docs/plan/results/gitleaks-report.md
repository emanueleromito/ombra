# Secret scan of the full history (T4.7)

- **Tool:** gitleaks v8.30.1 (release binary, linux x64), default rules plus the repo's `.gitleaks.toml`.
- **Scope:** every commit reachable from every ref of the local clone, 208 commits, about 2.6 MB.
- **Command:** `gitleaks detect --log-opts=--all --redact -r report.json -f json -s .` (the `detect` form of `gitleaks git --log-opts=--all`).
- **Hits without the allowlist:** 2. **Real secrets:** 0. **Hits with the allowlist:** 0.

| # | Rule | File | Commit | Verdict |
| --- | --- | --- | --- | --- |
| 1 | generic-api-key | `client/ombra/_proto.py` line 20 | `507b4226` | False positive. The matched text is the Python module name `ombra_api.v1.ombra_pb2` in a tuple of module names. No credential. |
| 2 | generic-api-key | `client/ombra/collection.py` line 171 | `4831d558` | False positive. The matched text is the keyword argument `master_key=m.master_key`, a variable reference, not a literal key. |

No secret value appears in either hit. Both are allowlisted in `.gitleaks.toml` by file path and an exact regex, so a real key in those files would still be caught. A rerun with the config reports "no leaks found".

The scan covers the history of the private repository. Rerun it on the exported public history (T4.8) before the first public push.
