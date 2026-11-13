# Encrypted search

After [expansion](query-expansion.md) the server holds the query as $D$ ciphertexts $\hat q_j$, each with $q_j$ in every slot. A stored block is $D$ ciphertexts $a_j$ with dimension $j$ of up to {{ preset("balanced", "batch_size") }} vectors (see [column packing](encoding.md)). The server now computes, for each block, one ciphertext whose slot $s$ is the score of vector $s$. This page walks through the four steps: multiply and add, relinearize, rescale, and what is left of the level budget.

## Multiply and add

The score of the vector in slot $s$ is

$$\text{score}[s] = \sum_{j=0}^{D-1} \hat q_j[s] \cdot a_j[s]$$

Multiplying two ciphertexts slot by slot is a multiplication of polynomials, and every ciphertext has two polynomials. A ciphertext $(c_0, c_1)$ decrypts as $c_0 + c_1 s$, where $s$ is the secret key. The product of two ciphertexts $(a_0, a_1)$ and $(b_0, b_1)$ is the product of those expressions:

$$(a_0 + a_1 s)(b_0 + b_1 s) = a_0 b_0 + (a_0 b_1 + a_1 b_0)\, s + a_1 b_1\, s^2$$

So the product has three components $(a_0 b_0,\ a_0 b_1 + a_1 b_0,\ a_1 b_1)$, and the sum over $j$ is the componentwise sum of such triples. The server never touches $s$. It only combines the ciphertext polynomials, in NTT form, one residue tower at a time (see [CKKS in simple terms](../concepts/ckks.md)), where polynomial multiplication is one modular multiplication per coefficient.

In the kernel, each block takes 4 modular multiplications and 3 modular additions per coefficient and dimension. In the balanced preset a fresh ciphertext has {{ preset("balanced", "multiplicative_depth") + 1 }} towers and $N = {{ preset("balanced", "ring_dim") }}$, so one block at $D = 768$ costs

$$768 \cdot 4 \cdot {{ preset("balanced", "multiplicative_depth") + 1 }} \cdot {{ preset("balanced", "ring_dim") }} = {{ 768 * 4 * (preset("balanced", "multiplicative_depth") + 1) * preset("balanced", "ring_dim") }}$$

modular multiplications, for {{ preset("balanced", "batch_size") }} vectors, which is {{ 768 * 4 * (preset("balanced", "multiplicative_depth") + 1) * preset("balanced", "ring_dim") // preset("balanced", "batch_size") }} per vector. Those are integer operations on 64-bit words and they map well to a GPU. The server runs a block as one kernel launch.

**Worked example.** Take the $d = 4$ block from the [column packing page](encoding.md) with slot 2 holding $v_2 = (0, 0, 0.6, 0.8)$, and $q = (0.5, -0.5, 0.5, 0.5)$. The plaintext values are $a_j[2] = 0, 0, 0.6, 0.8$ and $\hat q_j[2] = 0.5, -0.5, 0.5, 0.5$. With scale $\Delta$ on both sides, the encoded integers are about $a_j[2]\,\Delta$ and $\hat q_j[2]\,\Delta$, so each product is $a_j[2]\,\hat q_j[2]\,\Delta^2$ and the sum in slot 2 is

$$(0 + 0 + 0.3 + 0.4)\,\Delta^2 = 0.7\,\Delta^2$$

The result is correct, but it now sits at scale $\Delta^2$ and has three components.

## Relinearize

A three-component ciphertext decrypts with $s^2$ as well, and it is larger. Relinearization turns it back into two components that decrypt with $s$. It is a key switch: the $s^2$ part is re-expressed in terms of $s$ with a public evaluation key that the client uploaded during setup, which hides $s$ the way a public key does.

A key switch is much more expensive than one multiplication, and the sum above has $D$ products. Since relinearization is linear, doing it once on the sum gives the same result as doing it on each product, so the server relinearizes once per block, not once per dimension. At $D = 768$ that is one key switch against 768 multiplications.

## Rescale

After multiplication the scale is $\Delta^2$, which doubles the number of bits every value takes, and a second multiplication would overflow the modulus. Rescaling divides the ciphertext by the last modulus $q_\ell$ of its chain, which is close to $\Delta$, and drops that tower:

$$\text{scale} = \Delta^2 \;\longrightarrow\; \frac{\Delta^2}{q_\ell} \approx \Delta$$

In the example the encoded sum $0.7\,\Delta^2$ becomes about $0.7\,\Delta$ again. In the balanced preset the scaling moduli have {{ preset("balanced", "scaling_mod_size") }} bits and the first modulus has {{ preset("balanced", "first_mod_size") }} bits, so $\Delta \approx 2^{ {{ preset("balanced", "scaling_mod_size") }} }$. The ciphertext goes from {{ preset("balanced", "multiplicative_depth") + 1 }} towers to {{ preset("balanced", "multiplicative_depth") }}. This is one of the two levels the preset provides, as the next section explains. The server sends this ciphertext to the client, which decrypts it with the secret key and reads one score per slot.

## The level budget

A level is one rescale. The preset's `multiplicative_depth` sets how many a fresh ciphertext can take: {{ preset("balanced", "multiplicative_depth") }} in both product presets. The search above uses one of them. Everything else in the pipeline is free of levels:

| step | levels used |
| --- | --- |
| expansion (adds, subtractions, automorphisms, shifts) | 0 |
| multiply and add over $D$ columns | 0 (only grows the scale) |
| relinearize | 0 |
| rescale | 1 |
| tombstone penalty for deleted vectors (a plaintext added to the result) | 0 |

So a result is at level 1 of {{ preset("balanced", "multiplicative_depth") }}, with one level spare. The core library also has a variant that multiplies the scores by a 0/1 mask and rescales a second time, using the second level (`GpuSearch` in `core/`). The server does not use it. Filters are part of the query (extra columns and a penalty weight, see [filters](filters.md)), and deletes are a plaintext addition, so neither costs a level.

!!! deep-dive "How big can a score get"
    CKKS stores a value times $\Delta$ modulo the remaining modulus, so a slot that gets too large wraps around and decrypts to nonsense. The client allows a slot value of at most $2^{\,b}$, with

    $$b = \text{first\_mod\_size}-\text{scaling\_mod\_size}-1-3$$

    which is the bound at the lowest level minus a margin of 3 bits. For the balanced preset $b = {{ preset("balanced", "first_mod_size") - preset("balanced", "scaling_mod_size") - 4 }}$, so slots stay below {{ 2 ** (preset("balanced", "first_mod_size") - preset("balanced", "scaling_mod_size") - 4) }}. A score is at most 1 in size, and a filter adds a penalty of $\lambda = {{ const("ombra.presets.FILTER_LAMBDA") }}$ per matched field. That caps the number of filter fields at $F_{max} = {{ fmax("balanced") }}$ for the balanced preset and {{ fmax("fast") }} for the fast one.

## What decides the speed

Per block, the search is one kernel launch for the sum, then a relinearization and a rescale for a group of blocks. In the measured runs the kernel was 3.5% to 8.1% of server time at 1,000,000 vectors with 20% of them searched on an A100, so the [expansion](query-expansion.md) is the larger cost. Each GPU thread handles {{ cpp("core/kernels/column_search.cu", "kBlockTile") }} blocks at once, so the query words are read once for all of them.

Search only works on CKKS with fixed manual scaling and hybrid key switching, which both presets use. Other settings are refused. Every step is written to give exactly the words OpenFHE's CPU code gives, and the tests compare the two.

## Where the code is

The kernel and its memory layout are in [`core/kernels/column_search.cu`](https://github.com/erailabs/ombra/blob/main/core/kernels/column_search.cu), with the host side in [`core/include/ombra/core/column_search.hpp`](https://github.com/erailabs/ombra/blob/main/core/include/ombra/core/column_search.hpp). Relinearization and rescaling are in [`core/include/ombra/core/gpu_search.hpp`](https://github.com/erailabs/ombra/blob/main/core/include/ombra/core/gpu_search.hpp). The server loop that runs them for each block is in [`server/search/src/session.cpp`](https://github.com/erailabs/ombra/blob/main/server/search/src/session.cpp).
