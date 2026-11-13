# Encoding and column packing

A CKKS ciphertext holds a long list of numbers, one per slot (see [CKKS in simple terms](../concepts/ckks.md)). The balanced preset has {{ preset("balanced", "batch_size") }} slots per ciphertext and the fast preset has {{ preset("fast", "batch_size") }}. The question is what to put in those slots. Ombra puts one dimension of many vectors in each ciphertext. This page explains why, and what it costs.

## Row packing and column packing

The obvious layout is row packing: one vector per ciphertext, its $d$ values in the first $d$ slots. A dot product then needs a sum across slots. CKKS can only add slots together by rotating the ciphertext and adding it to itself, about $\log_2 d$ times for each vector. A rotation is a key switch, the most expensive operation in the scheme, so scoring a million vectors would take millions of them.

Column packing turns the layout by 90 degrees. Take $S$ vectors $v_0, \dots, v_{S-1}$ with $d$ values each, where $S$ is the number of slots. Ciphertext $a_j$ holds dimension $j$ of all of them:

$$a_j[s] = v_s[j] \qquad 0 \le j < d,\ 0 \le s < S$$

Here $a_j[s]$ is slot $s$ of ciphertext $a_j$ and $v_s[j]$ is value $j$ of vector $s$. The set $a_0, \dots, a_{d-1}$ is one block: $d$ ciphertexts that together hold $S$ vectors.

Now take a query $q$ with $d$ values, and ciphertexts $\hat q_j$ that hold $q_j$ in every slot. Multiplying slot by slot and summing over $j$ gives the dot product of $q$ with every vector at once:

$$\text{score}[s] = \sum_{j=0}^{d-1} \hat q_j[s] \cdot a_j[s] = \sum_{j=0}^{d-1} q_j \, v_s[j] = \langle q, v_s \rangle$$

Only multiplications and additions appear, no rotations, and slot $s$ of the result holds the complete score of vector $s$. Getting the $\hat q_j$ from a single query ciphertext is the job of [query expansion](query-expansion.md). The multiply-and-add itself is on the [encrypted search](encrypted-search.md) page.

### Worked example

Take $d = 4$ and a toy block of $S = 4$ vectors (real blocks have thousands). Each row is a vector, and the server stores the columns:

| vector | $v[0]$ | $v[1]$ | $v[2]$ | $v[3]$ |
| --- | --- | --- | --- | --- |
| $v_0$ | 0.5 | 0.5 | 0.5 | 0.5 |
| $v_1$ | 0.5 | -0.5 | 0.5 | 0.5 |
| $v_2$ | 0 | 0 | 0.6 | 0.8 |
| $v_3$ | -0.5 | 0.5 | -0.5 | -0.5 |

$$a_0 = [0.5,\ 0.5,\ 0,\ -0.5] \quad a_1 = [0.5,\ -0.5,\ 0,\ 0.5] \quad a_2 = [0.5,\ 0.5,\ 0.6,\ -0.5] \quad a_3 = [0.5,\ 0.5,\ 0.8,\ -0.5]$$

Let the query be $q = (0.5, -0.5, 0.5, 0.5)$. After expansion the server has $\hat q_0 = [0.5, 0.5, 0.5, 0.5]$, $\hat q_1 = [-0.5, -0.5, -0.5, -0.5]$, and so on. Multiplying and adding gives

$$\text{score} = [\,0.5,\ 1.0,\ 0.7,\ -1.0\,]$$

For slot 2, which is $v_2$: $0.5 \cdot 0 + (-0.5) \cdot 0 + 0.5 \cdot 0.6 + 0.5 \cdot 0.8 = 0.7$. Four dot products came out of 4 multiplications and 3 additions on whole ciphertexts. Vectors are normalized to unit length before encryption, so these dot products are cosine similarities, and $v_1 = q$ scores 1.0.

## What a block looks like in practice

A block of the balanced preset holds up to {{ preset("balanced", "batch_size") }} vectors. At $d = 768$ it is 768 ciphertexts. If a cluster has fewer vectors than slots, the unused slots are zero and the block costs the same as a full one, so small clusters are expensive per vector. A cluster of $n$ vectors uses $\lceil n / S \rceil$ blocks.

Each ciphertext has two polynomials, each with one residue tower per modulus (see [CKKS in simple terms](../concepts/ckks.md)), and every coefficient is a 64-bit word. A fresh column in the balanced preset has {{ preset("balanced", "multiplicative_depth") + 1 }} towers and ring dimension $N = {{ preset("balanced", "ring_dim") }}$, so on the GPU it takes

$$2 \cdot {{ preset("balanced", "multiplicative_depth") + 1 }} \cdot {{ preset("balanced", "ring_dim") }} \cdot 8 \text{ bytes} = {{ 2 * (preset("balanced", "multiplicative_depth") + 1) * preset("balanced", "ring_dim") * 8 // 1024 }} \text{ KiB}.$$

A block of 768 columns is then about {{ 2 * (preset("balanced", "multiplicative_depth") + 1) * preset("balanced", "ring_dim") * 8 * 768 // 1024 // 1024 }} MiB of GPU memory for {{ preset("balanced", "batch_size") }} vectors, around {{ 2 * (preset("balanced", "multiplicative_depth") + 1) * preset("balanced", "ring_dim") * 8 * 768 // preset("balanced", "batch_size") // 1024 }} KiB per vector, against 3 KiB for 768 plain 32-bit floats. That is the price of keeping everything encrypted. The [storage page](../performance/storage.md) covers stored sizes.

Some dimensions do not fit. The query travels as one ciphertext that uses $2d$ of its $N$ coefficients (see [query expansion](query-expansion.md)), so $d \le N/2$. That is {{ preset("fast", "ring_dim") // 2 }} for the fast preset and {{ preset("balanced", "ring_dim") // 2 }} for the balanced one. The client accepts the dimensions {{ const("ombra.presets.SUPPORTED_DIMENSIONS") }} and refuses others.

### Filter columns

A filter field with $V$ possible values adds $V$ more columns to the block, one per value, holding 1 in the slots of the vectors that have that value and 0 elsewhere. The query then carries $d + \sum V$ entries and the search treats them all alike. A block therefore has $D = d + \sum V$ columns, and the cost of a search grows with $D$. How the filter entries are used is on the [filters page](filters.md).

## How the client builds a column

The client takes value $j$ of each vector in the block, pads with zeros up to the slot count, sets any value smaller than $10^{-9}$ to zero (OpenFHE refuses to encode denormal-sized numbers), and passes the list to OpenFHE's `MakeCKKSPackedPlaintext` and `Encrypt`. Encoding and encryption are OpenFHE's. Ombra only decides which number goes in which slot.

!!! deep-dive "Why not a few dimensions per ciphertext"
    A middle layout puts a few dimensions of one vector into each ciphertext. The query then needs fewer ciphertexts, but each dot product needs rotations again. With column packing the rotations happen once per query, during expansion, and not once per block. The cost is that expansion has to make $D$ ciphertexts out of one, which is the largest server step when a search touches few blocks.

The GPU code that reads these columns is [`core/include/ombra/core/column_search.hpp`](https://github.com/erailabs/ombra/blob/main/core/include/ombra/core/column_search.hpp), with the kernel in [`core/kernels/column_search.cu`](https://github.com/erailabs/ombra/blob/main/core/kernels/column_search.cu). The client builds the columns in `Collection._columns` and `KeyKit.encrypt_column` in [`client/ombra/collection.py`](https://github.com/erailabs/ombra/blob/main/client/ombra/collection.py).
