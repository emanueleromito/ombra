# Presets

A preset fixes the encryption parameters. You pick a name and a vector dimension. You never pick ring sizes or moduli, and every preset is at least 128-bit secure: the client refuses to build one that is not. The values below are read from the code when the site is built.

| Preset | Ring dimension $N$ | Slots | Multiplicative depth | Filter fields $F_{max}$ |
| --- | --- | --- | --- | --- |
| `fast` | {{ preset("fast", "ring_dim") }} | {{ preset("fast", "batch_size") }} | {{ preset("fast", "multiplicative_depth") }} | {{ fmax("fast") }} |
| `balanced` | {{ preset("balanced", "ring_dim") }} | {{ preset("balanced", "batch_size") }} | {{ preset("balanced", "multiplicative_depth") }} | {{ fmax("balanced") }} |

The filter penalty weight is $\lambda = {{ const("ombra.presets.FILTER_LAMBDA") }}$.

## Details

The same presets with their lower-level parameters. Bit sizes are those of the RNS moduli: the first modulus holds the final result, and each scaling modulus matches the scale $\Delta = 2^{\text{scaling bits}}$.

| Preset | Core name | Security level | First modulus bits | Scaling modulus bits | Scaling technique | Key switching |
| --- | --- | --- | --- | --- | --- | --- |
| `fast` | `{{ const("ombra.presets.PRODUCT_PRESETS")["fast"] }}` | {{ preset("fast", "security_level") }} | {{ preset("fast", "first_mod_size") }} | {{ preset("fast", "scaling_mod_size") }} | {{ preset("fast", "scaling_technique") }} | {{ preset("fast", "key_switch_technique") }} |
| `balanced` | `{{ const("ombra.presets.PRODUCT_PRESETS")["balanced"] }}` | {{ preset("balanced", "security_level") }} | {{ preset("balanced", "first_mod_size") }} | {{ preset("balanced", "scaling_mod_size") }} | {{ preset("balanced", "scaling_technique") }} | {{ preset("balanced", "key_switch_technique") }} |

The core name is what a collection stores and what the server checks against the envelope of every ciphertext. `HEStd_128_classic` is OpenFHE's 128-bit security level. See [Parameters](../security/parameters.md) for the security argument.

The default preset is `{{ const("ombra.presets.DEFAULT_PRESET") }}` against a server. In local mode, `create_collection` defaults to `fast`. A third name, `precise`, is reserved and refused for now.

## Dimensions

Supported vector dimensions are {{ const("ombra.presets.SUPPORTED_DIMENSIONS") | join(", ") }}. The dimension plus the sum of the filter fields' value counts must not exceed half the ring dimension, since the query is packed into one ciphertext. Slots are the number of vectors one ciphertext holds, so a block of one cluster has at most that many vectors.

## Filter limits

A collection can have at most {{ const("ombra.presets.MAX_FILTER_FIELDS") }} filter fields of at most {{ const("ombra.presets.MAX_FILTER_CARDINALITY") }} values each. The preset can lower the field limit, because filtering adds $\lambda$ times a count to every score, and the result must stay within what the lowest level of the ciphertext can hold. $F_{max}$ is that limit:

$$F_{max} = \min\left(16,\ \left\lfloor \frac{2^{b-3} - 1}{\lambda} \right\rfloor - 1\right), \qquad b = \text{first bits} - \text{scaling bits} - 1$$

where $\lambda$ is the penalty weight above and the 3 is a safety margin in bits. A tombstoned slot reaches $1 + \lambda (F + 1)$, which is what the bound protects.

{% set b = preset("balanced", "first_mod_size") - preset("balanced", "scaling_mod_size") - 1 %}
For `balanced`: $b = {{ preset("balanced", "first_mod_size") }} - {{ preset("balanced", "scaling_mod_size") }} - 1 = {{ b }}$, so $2^{b-3} = {{ 2 ** (b - 3) }}$, and $F_{max} = {{ fmax("balanced") }}$.

Where each number comes from in the code: `client/ombra/resources/ckks_presets.json` holds the parameters (the server reads the same file at build time), and `client/ombra/presets.py` derives the layout and $F_{max}$.
