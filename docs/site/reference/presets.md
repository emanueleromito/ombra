# Presets

The parameters of each preset. The values below are read from the code when the site is built.

| Preset | Ring dimension $N$ | Slots | Multiplicative depth | Filter fields $F_{max}$ |
| --- | --- | --- | --- | --- |
| `fast` | {{ preset("fast", "ring_dim") }} | {{ preset("fast", "batch_size") }} | {{ preset("fast", "multiplicative_depth") }} | {{ fmax("fast") }} |
| `balanced` | {{ preset("balanced", "ring_dim") }} | {{ preset("balanced", "batch_size") }} | {{ preset("balanced", "multiplicative_depth") }} | {{ fmax("balanced") }} |

The filter penalty weight is $\lambda = {{ const("ombra.presets.FILTER_LAMBDA") }}$.
