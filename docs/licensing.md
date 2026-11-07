# Licensing decision

_Decided: 2026-10-31_

## Summary

The FHE vector DB is **source-available**: the full code is public on GitHub, free for individuals and non-commercial organizations, and **any company using it for business, including internally for its own AI, must buy a commercial license**. The base license is **PolyForm Noncommercial 1.0.0**; companies sign a separate commercial license agreement. Because it restricts commercial use, we never call it "open source".

## Context

The product is a vector database where both the stored vectors and the search queries stay encrypted with fully homomorphic encryption (FHE), so the server never sees the data. Its buyers are mainly regulated companies (healthcare, finance, legal, public sector) that can't use normal vector databases for sensitive data.

We want two things that pull against each other:

- **Trust and adoption:** security buyers need to audit the cryptography, so the code should be public.
- **Revenue:** companies that get business value from it should pay, even if they only use it internally.

Today, no production-grade FHE vector DB is publicly available as source code. The main commercial one (CryptoLab's enVector) is proprietary.

## Decision

We release the code under **PolyForm Noncommercial 1.0.0** and sell a separate **commercial license** to companies.

| Who | Uses it free? | What they need |
| --- | --- | --- |
| Individuals, hobbyists, students | Yes | Nothing |
| Universities and research institutes | Yes | Nothing |
| Charities, public bodies, non-commercial organizations | Yes | Nothing |
| Any company using it for business (internal AI, products, services) | No | Paid commercial license |
| Companies evaluating it before buying | Yes, for a limited trial | Free evaluation license |

The commercial license replaces the PolyForm terms for that company. Pricing to start: a yearly license per company or per production cluster, plus an optional paid support tier.

**Fallback option:** if we later want small startups to use it for free, we switch to **PolyForm Small Business** (free for companies under about 100 people and about $1M revenue).

## How it works in practice

1. **GitHub repo:** full source code public, with the PolyForm Noncommercial `LICENSE` file and a separate `COMMERCIAL.md` explaining how companies buy a license.
2. **README wording:** "Free for personal, academic and non-commercial use. Companies need a commercial license." Never "open source"; use "source-available".
3. **Contributor License Agreement (CLA):** every outside contributor signs one (e.g. via CLA Assistant on GitHub), so we can include their code in commercial licenses.
4. **Commercial license agreement:** a standard contract for companies, covering scope (per company or per cluster), duration, support, and an annual audit clause.
5. **Trademark:** register the product name, so copies or forks can't use it.
6. **Legal entity:** needed to sign contracts and receive payments (in Switzerland, typically a GmbH).

## Tracking usage and enforcement

We rely mainly on companies respecting the license; large regulated companies usually do. To find and convert those that don't:

- **Download gate:** free account with name, email, company and intended use before downloading.
- **License keys:** every install activates with a key (free non-commercial or paid commercial); offline signed license files for air-gapped customers.
- **Minimal, transparent telemetry:** only license key, version and rough usage numbers. Never vectors, queries or customer data.
- **Package download analytics:** tools like Scarf show which organizations pull the Docker or pip packages.
- **Public signals:** job posts, talks, blog posts and support emails from company domains.

When we find unlicensed commercial use, the first step is a friendly sales email, not a legal threat.
