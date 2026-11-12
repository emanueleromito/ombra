# 0001. Source-available license

- **Status:** accepted
- **Date:** 2026-10-31

## Context

Security buyers need to audit the cryptography, so the code must be public. But companies using it for business, even only internally, should pay. No real open source license allows "companies must pay for internal use".

## Options considered

- **A. AGPL + commercial license.** True open source, but companies can use it internally, unmodified, for free.
- **B. BSL / FSL / Elastic License.** Mainly block competing hosted services, not internal use.
- **C. Freeware (closed binaries + license keys).** Companies pay, but nobody can audit the crypto.
- **D. PolyForm Noncommercial + commercial license.** Code public; free for individuals and non-commercial organizations; any business use needs a paid license.

## Decision

D. PolyForm Noncommercial 1.0.0 in `LICENSE`, commercial terms in `LICENSING.md`. Fallback if we want small startups free later: PolyForm Small Business.

## Consequences

We never call the project "open source". Outside contributors must sign a CLA. Full reasoning in [licensing.md](../licensing.md).
