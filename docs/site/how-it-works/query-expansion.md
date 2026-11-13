# Query expansion

[Column packing](encoding.md) needs the query as $D$ ciphertexts, one per dimension, each with the same value $q_j$ in every slot. Sending that would cost the client $D$ encryptions and about $D$ times the upload. Instead the client sends one ciphertext and the server expands it. This page shows how, why the expansion uses no multiplication level, and where the factor $2^{-L}$ in the client code comes from.

## The query as a polynomial

A CKKS plaintext is a polynomial of degree below $N$ with coefficients in the ring $\mathbb{Z}[X]/(X^N + 1)$, in which $X^N = -1$. The query is encoded so that its coefficients, not its slots, carry the query values. For a query $q_0, \dots, q_{d-1}$ the client builds

$$p(X) = q_0 + \sum_{j=1}^{d-1} q_j \left( X^j + X^{ -j } \right)$$

Since $X^N = -1$, we get $X^{ -j } = -X^{N-j}$. So coefficient $j$ of $p$ is $q_j$, coefficient $N-j$ is $-q_j$, and every other coefficient is zero. This symmetric shape makes the slot values real, which is what OpenFHE's default mode keeps. The client computes the slot values with a numpy FFT and hands them to OpenFHE for encoding and encryption.

**Worked example.** Take $d = 4$, $N = 16$ and $q = (0.5, -0.5, 0.5, 0.5)$. Then

$$p = 0.5-0.5X + 0.5X^2 + 0.5X^3-0.5X^{13}-0.5X^{14} + 0.5X^{15}$$

Coefficients 13, 14 and 15 are $-q_3$, $-q_2$ and $-q_1$, in that order. Slot $k$ of the plaintext is the value of $p$ at $\zeta^{5^k}$, where $\zeta = e^{i\pi/N}$ is a root of unity, and for a symmetric $p$ this is

$$z_k = q_0 + 2\sum_{j=1}^{d-1} q_j \cos\!\left( \frac{\pi \, j \, 5^k}{N} \right)$$

For $k = 0$ this is $0.5 + 2(-0.5\cos\frac{\pi}{16} + 0.5\cos\frac{2\pi}{16} + 0.5\cos\frac{3\pi}{16}) = 1.2746$. The next slots are $-1.419$, $-1.1745$, $-0.1439$ and so on. There are $N/2 = 8$ of them, and they are real. (The client actually encodes $p$ scaled by $2^{-L}$, so the numbers it stores are 8 times smaller. More on that below.)

## One split step

The server needs one ciphertext per $q_j$, each holding only the constant $q_j$ (a constant polynomial has the same value in every slot). It gets there by repeatedly splitting a polynomial into its even and odd part, using an automorphism.

The automorphism $\tau_t$ replaces $X$ by $X^t$ for an odd $t$. It moves coefficient $i$ to position $it \bmod 2N$, with a sign flip when that position is past $N$. On ciphertexts it is a key switch with a public Galois key, so the server computes it without seeing anything.

Take $t = N/2^l + 1$ and a polynomial whose nonzero coefficients sit at multiples of $2^l$, so $i = 2^l m$. Then $it = i + mN$, and $X^{it} = (-1)^m X^i$ because $X^N = -1$. The automorphism flips the sign of the coefficients with odd $m$ and leaves the others alone. Adding $c$ and $\tau_t(c)$ keeps the coefficients with even $m$, doubled. Subtracting keeps the odd ones, doubled. So with $c$ a node at depth $l$:

$$\text{even child} = c + \tau_t(c) \qquad \text{odd child} = \left( c - \tau_t(c) \right) \cdot X^{ -2^l }$$

The multiplication by $X^{ -2^l }$ moves the odd coefficients down by $2^l$ so that both children again have their coefficients at multiples of $2^{l+1}$. Both children hold twice the coefficients they took over, which is where the factor 2 per level comes from.

## The expansion tree

Start with the root $p$ at depth 0. Apply the split with $l = 0$, then to both children with $l = 1$, and so on. After $L$ levels each leaf holds the coefficients whose index has one particular residue modulo $2^L$, shifted to start at 0. We want leaf $r$ to be the constant $q_r$, which needs $L$ large enough that only coefficient $r$ falls in residue $r$:

$$L = \lceil \log_2 (2d) \rceil$$

This is `expansion_depth(d)` in the client and the server, and it is the smallest $L$ with $2^L \ge 2d$. Using $2d$ and not $d$ keeps coefficient $N-j$ (holding $-q_j$) out of the residue of $r$.

!!! deep-dive "Why $2^L \ge 2d$ is enough"
    Coefficients of $p$ sit at indices $j \in [0, d-1]$ and $N-j$ for $j \in [1, d-1]$. Leaf $r$ collects the indices congruent to $r$ modulo $2^L$. From the first group only $j = r$ qualifies, since $d \le 2^L$. The second group qualifies when $N-j \equiv r$. As $2^L$ divides $N$ (the code checks $d \le N/2$, so $2^L \le N$), this means $j + r \equiv 0$ modulo $2^L$. But $2 \le j + r \le 2d-2 < 2^L$ for $r \ge 1$, and for $r = 0$ we have $j \ge 1$ and $j < 2^L$, so it never holds. Leaf $r$ therefore holds $2^L q_r$ and nothing else.

The tree only creates the branches it needs. A node with label $r$ makes an odd child with label $r + 2^l$ only when $r + 2^l < d$. So exactly $d$ leaves come out, all at depth $L$.

**Worked example.** For $d = 4$ and $N = 16$ the depth is $L = \lceil \log_2 8 \rceil = 3$ and the three splits use $t = 17, 9, 5$. These are the Galois elements the client prepares keys for. Write $q = (0.5, -0.5, 0.5, 0.5)$ and scale by $2^{-3}$, so $p$ has coefficients $\pm 0.0625$. Then (showing only the nonzero coefficients):

| depth | node (label) | polynomial |
| --- | --- | --- |
| 0 | root | $0.0625-0.0625X + 0.0625X^2 + 0.0625X^3-0.0625X^{13}-0.0625X^{14} + 0.0625X^{15}$ |
| 1 | 0 | $0.125 + 0.125X^2-0.125X^{14}$ |
| 1 | 1 | $-0.125 + 0.125X^2-0.125X^{12} + 0.125X^{14}$ |
| 2 | 0 | $0.25$ |
| 2 | 2 | $0.25-0.25X^{12}$ |
| 2 | 1 | $-0.25-0.25X^{12}$ |
| 2 | 3 | $0.25 + 0.25X^{12}$ |
| 3 | 0, 1, 2, 3 | $0.5,\ -0.5,\ 0.5,\ 0.5$ |

At depth 1 the label-0 node is even child of the root: $c + \tau_{17}(c)$ flips the sign of odd coefficients, so the sum keeps coefficients 0, 2 and 14 doubled. The label-1 child is $(c-\tau_{17}(c))$ shifted down by one. At the last split, node 2 holds $0.25-0.25X^{12}$: $\tau_5$ sends $X^{12}$ to $X^{60}$, and $X^{60} = (X^{16})^3 X^{12} = -X^{12}$, so $\tau_5(c) = 0.25 + 0.25X^{12}$ and the sum is $0.5$. The odd children of the last level would get labels 4 to 7, which are not below $d = 4$, so they are never made. The leaves are the constants $0.5, -0.5, 0.5, 0.5$, which is $q$ again.

The tree has 1, 2, 4 and 4 nodes at depths 0 to 3, so the example takes 1 + 2 + 4 = 7 splits and 7 automorphisms. Splits that only produce an even child still cost a key switch, so the number of automorphisms is more than $d-1$. At $d = 768$ the depth is $L = {{ (2 * 768-1).bit_length() }}$, and the number of automorphisms is the sum over depths of $\min(2^l, d)$:

{% set ns = namespace(total=0) %}{% for l in range((2 * 768-1).bit_length()) %}{% set ns.total = ns.total + ([2 ** l, 768] | min) %}{% endfor %}
$$\sum_{l=0}^{ {{ (2 * 768-1).bit_length()-1 }} } \min(2^l, 768) = {{ ns.total }}$$

All splits at one depth use the same Galois element and the same shift, so the GPU runs a whole depth in one batch.

## Why it uses no level

In CKKS, multiplying by a number such as $2^{-L}$ is a multiplication, and every multiplication costs a level at the next rescale (see [CKKS in simple terms](../concepts/ckks.md)). Nothing in the tree does that:

- Addition and subtraction of ciphertexts at the same level leave the scale and the level alone.
- The multiplication by $X^{ -2^l }$ is an exact permutation of coefficients with sign changes. It changes no scale.
- The automorphism with a Galois key switches the key and adds a little noise. It does not drop a modulus.

The tree does leave one unwanted factor: each leaf is $2^L$ times the value we want. Removing it would take a multiplication by $2^{-L}$, and so a level. Instead the client divides by $2^L$ before encrypting. Scaling a number by a power of two changes only its floating-point exponent, so $q \cdot 2^{-L}$ is exact in 64-bit floats, and the encrypted polynomial has coefficients $q_j 2^{-L}$. After $L$ doublings the leaf is exactly $q_j$. That is `prescale_query` in the client. The norm check on the query happens before the scaling, on the real $q$.

!!! deep-dive "What the pre-scaling costs"
    Pre-scaling shrinks the signal but not the noise: the encryption noise of the query ciphertext is the same as for any fresh ciphertext, and every split adds the noise of a key switch. Each split also doubles what it passes on, so a leaf can carry up to about $2^L$ times the noise of a fresh ciphertext while holding $q_j$ at its true size. The leaves are therefore noisier than freshly encrypted ones. The effect on search quality is measured on the [accuracy page](accuracy.md).

## What is slow

Expansion is the biggest server step when a search touches only a few blocks. Measured on an A100 at 100,000 vectors with 5% of them searched, it took 62% to 74% of the server time. The reason is the key switches: at $d = 768$ there are {{ ns.total }} of them per query, one per split, and the cost does not depend on how many blocks are searched. The memory is also large. Every leaf is a full ciphertext, and the tree keeps two consecutive depths alive, so at $d = 768$ the largest point is $768 + 768 = {{ 2 * 768 }}$ ciphertexts, which at {{ 2 * (preset("balanced", "multiplicative_depth") + 1) * preset("balanced", "ring_dim") * 8 // 1024 }} KiB each in the balanced preset is about {{ 2 * 768 * 2 * (preset("balanced", "multiplicative_depth") + 1) * preset("balanced", "ring_dim") * 8 // 1024 // 1024 }} MiB of GPU memory per query.

Expansion needs one Galois key per depth, $L$ keys in total, which the client generates and uploads as public evaluation keys. The query dimension is limited to $D \le N/2$.

## Where the code is

The tree is in [`core/include/ombra/core/expansion_tree.hpp`](https://github.com/erailabs/ombra/blob/main/core/include/ombra/core/expansion_tree.hpp). It is generic over the operations that do the splits, so the same tree runs on the CPU reference in [`core/src/openfhe_expand_ops.hpp`](https://github.com/erailabs/ombra/blob/main/core/src/openfhe_expand_ops.hpp) (OpenFHE's `EvalAutomorphism`, `EvalAdd`, `EvalSub` and an exact shift) and on the GPU in [`core/include/ombra/core/expansion_device.hpp`](https://github.com/erailabs/ombra/blob/main/core/include/ombra/core/expansion_device.hpp). The query encoding is in [`client/ombra/query.py`](https://github.com/erailabs/ombra/blob/main/client/ombra/query.py).
