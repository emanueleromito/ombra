# Security policy

## Reporting a vulnerability

Please report vulnerabilities privately. Use GitHub private vulnerability reporting: open the Security tab of this repository and choose "Report a vulnerability". Do not open a public issue or pull request for a security problem.

If you cannot use GitHub, write to SECURITY_CONTACT_EMAIL_PLACEHOLDER (to be replaced with the project's security address before the first public release).

Include the version or commit, what you did, what you expected and what happened. A short proof of concept helps. Use synthetic data only; do not send real vectors, queries, documents or keys.

## What to expect

- We acknowledge your report within 3 working days.
- We send an assessment (confirmed or not, severity, planned fix) within 10 working days.
- We fix confirmed issues in a patch release and credit you in the release notes unless you prefer not to be named. We ask you to keep the details private until the fix is out.

## Supported versions

| Version | Supported |
| --- | --- |
| 0.1.x | yes |
| older | no |

## Scope

In scope: the `ombra` Python client, the server (gRPC API, storage, scheduling), the CUDA and C++ code in `core/`, the integrations, and the build and release files in this repository. Examples: a way to recover the secret key or plaintext from what the server sees, FHE parameters below 128-bit security, a flaw in the ciphertext or key file formats, authentication or TLS bypass, memory-safety bugs in code that parses untrusted input.

Out of scope: weaknesses in OpenFHE itself (report those to the OpenFHE project), leakage the documented threat model already states (for example what the server learns from access patterns and result sizes), denial of service by a client that holds valid credentials and sends large but valid requests, and findings that need a compromised client machine.
