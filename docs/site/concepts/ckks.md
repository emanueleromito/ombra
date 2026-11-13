# CKKS in simple terms

CKKS is the homomorphic scheme Ombra uses. It works on real numbers, not integers, and it gives approximate answers. This page defines the six ideas the rest of the docs rely on: slots, scale, levels, rescaling, RNS towers and noise. Each one has a toy example with a ring dimension of $N = 16$. Real presets are much bigger, and the page says where.

## Slots and the ring

A ciphertext does not hold one number. It holds a vector of numbers, called **slots**, and every operation acts on all slots at once. Adding two ciphertexts adds slot by slot. Multiplying them multiplies slot by slot.

Under the hood a ciphertext is made of polynomials of degree below $N$, where $X^N = -1$. The number $N$ is the **ring dimension**. A polynomial with $N$ real coefficients can carry $N/2$ numbers in its slots, so with $N = 16$ there are 8 slots. The encoder turns 8 numbers into a polynomial so that multiplying two polynomials multiplies their slot values one by one.

Example with $N = 16$ and 8 slots:

$$
(1, 2, 3, 4, 5, 6, 7, 8) \cdot (2, 2, 2, 2, 0, 0, 0, 0) = (2, 4, 6, 8, 0, 0, 0, 0)
$$

The server does this with one ciphertext multiplication, not eight. Unused slots hold zero.

Ombra's presets use a much larger ring. `fast` has $N = {{ preset("fast", "ring_dim") }}$ and {{ preset("fast", "batch_size") }} slots. `balanced` has $N = {{ preset("balanced", "ring_dim") }}$ and {{ preset("balanced", "batch_size") }} slots. Each slot holds the value of one stored vector, which is why Ombra scores thousands of vectors with a handful of ciphertext operations.

### Rotations and automorphisms

Slots can be moved. Replacing $X$ with $X^k$ in a polynomial, for an odd $k$, is an **automorphism**. It keeps addition and multiplication intact and moves the data. With $N = 16$ and $X^{16} = -1$, take $p(X) = 1 + 2X + 3X^6$ and $k = 3$:

$$
p(X^3) = 1 + 2X^3 + 3X^{18} = 1 + 2X^3 + (-3)X^2
$$

because $X^{18} = X^{16} \cdot X^2 = -X^2$. The coefficients move and may change sign. For one special choice, $k = 5$, the effect on the slots is a rotation by one place: $(1, 2, 3, 4, 5, 6, 7, 8)$ becomes $(2, 3, 4, 5, 6, 7, 8, 1)$. Powers of 5 rotate by more places. Each automorphism needs its own key (see [Keys](keys.md)).

## Scale

CKKS stores real numbers as integers. To store 3.14 it multiplies by a fixed factor $\Delta$, the **scale**, and rounds. Toy example with $\Delta = 100$:

$$
3.14 \to 314, \qquad 2.00 \to 200
$$

Adding gives $314 + 200 = 514$, which reads back as $5.14$. Both numbers must carry the same scale to be added. Adding $314$ (scale 100) to a number stored at scale $10^4$ would give nonsense.

Multiplying is where the scale grows. $314 \cdot 200 = 62800$, and the result carries scale $\Delta^2 = 10^4$, so it reads back as $62800 / 10^4 = 6.28$. That is correct, but the number is now twice as wide in bits. Left alone, the scale squares with every multiplication and soon overflows the modulus. Rescaling fixes it.

In Ombra, $\Delta$ is about $2^{ {{ preset("balanced", "scaling_mod_size") }} }$ for `balanced` and $2^{ {{ preset("fast", "scaling_mod_size") }} }$ for `fast`. A value like 0.123 is stored as roughly $0.123 \cdot 2^{ {{ preset("balanced", "scaling_mod_size") }} }$, an integer of about {{ preset("balanced", "scaling_mod_size") }} bits.

## Levels and rescaling

**Rescaling** divides a ciphertext by about $\Delta$ and drops that factor from the modulus, which brings the scale back down. Continuing the example, $62800$ at scale $10^4$ is divided by $\Delta = 100$ and gives $628$ at scale $100$. It still reads as $6.28$, and it is back to the narrow width.

Each rescale uses up one **level**. The ciphertext modulus is built from a fixed chain of numbers, and a rescale removes the last one for good. When only the first is left, no more multiplication is possible. The number of multiplications you can chain is the **multiplicative depth**. Both Ombra presets have depth {{ preset("balanced", "multiplicative_depth") }}.

```mermaid
flowchart LR
    A["Fresh: 3 numbers in the chain"] -- "multiply, rescale" --> B["Level 1: 2 numbers"]
    B -- "multiply, rescale" --> C["Level 2: 1 number, no more multiplications"]
```

The chain for `balanced` is one number of {{ preset("balanced", "first_mod_size") }} bits and {{ preset("balanced", "multiplicative_depth") }} numbers of {{ preset("balanced", "scaling_mod_size") }} bits. `fast` uses {{ preset("fast", "first_mod_size") }} and {{ preset("fast", "scaling_mod_size") }} bits. The 50-bit numbers are what a rescale divides by, so they are chosen close to $\Delta$. The first number is bigger, because it has to hold the final result after the last rescale.

Additions do not use a level. Multiplying by a plain constant does, like any multiplication. A computation that needs more multiplications than the depth cannot run. Ombra has no bootstrapping to refresh levels, so each search is designed to fit in {{ preset("balanced", "multiplicative_depth") }}.

!!! deep-dive "How much room is left at the lowest level"
    After the last rescale only the first number, $q_0$, remains, and a stored value $v$ is the integer $v \cdot \Delta$. It must stay below $q_0 / 2$. For `balanced`, $q_0$ has {{ preset("balanced", "first_mod_size") }} bits and $\Delta$ has {{ preset("balanced", "scaling_mod_size") }}, so the value $v$ has to stay under $2^{ {{ preset("balanced", "first_mod_size") - preset("balanced", "scaling_mod_size") - 1 }} }$ = {{ 2 ** (preset("balanced", "first_mod_size") - preset("balanced", "scaling_mod_size") - 1) }}. Past that, the result wraps around and decrypts to garbage. Ombra keeps scores and filter penalties well below this limit and derives the number of filter fields it allows from it.

## RNS towers

The chain above is made of numbers about 50 bits long, but the total modulus $Q$ is their product: over 150 bits. Computers have no native type for that. The **residue number system (RNS)** avoids big integers by keeping a number as its remainders modulo each chain number, one **tower** per number. Each remainder fits in a 64-bit word, and arithmetic acts on every tower independently, which is what a GPU is good at.

Toy example with three small towers, $Q = 5 \cdot 7 \cdot 11 = 385$. The number 123 is stored as its remainders:

$$
123 \to (123 \bmod 5,\ 123 \bmod 7,\ 123 \bmod 11) = (3, 4, 2)
$$

Adding 200, stored as $(0, 4, 2)$, means adding tower by tower, each modulo its own number:

$$
(3, 4, 2) + (0, 4, 2) = (3,\ 8 \bmod 7,\ 4) = (3, 1, 4)
$$

and indeed $123 + 200 = 323$ has remainders $(3, 1, 4)$. Multiplication works the same way: $12 \cdot 30 = 360$ gives $(2,5,1) \cdot (0,2,8) = (0,\ 10 \bmod 7,\ 8) = (0, 3, 8)$, and $360$ has remainders $(0, 3, 8)$.

Dropping a level means dropping a tower. In the real presets each polynomial has {{ preset("balanced", "multiplicative_depth") + 1 }} towers when fresh, one for the first number and one per level, and one fewer after each rescale. A ciphertext gets smaller as it goes, so later operations are cheaper.

## Noise

Encryption hides a message by adding a small random error to it. That error is **noise**. It is what makes the ciphertext safe, since without it the secret key could be solved for, and it is also why the answer is approximate. Decrypting gives the message plus the noise.

Toy example, $\Delta = 100$. Encrypting $3.14$ stores $314$ plus a noise of, say, $+1$, so the ciphertext holds $315$. Decrypted, it reads $3.15$. The error is $0.01$, one unit in the last place of the scale.

Noise grows with every operation. Multiply this ciphertext by an exact $2.00$:

$$
315 \cdot 200 = 63000 \to \text{rescale} \to 630 \to 6.30
$$

The exact answer is $6.28$. The error is now $0.02$: it doubled, because the noise of $+1$ was multiplied by $2.00$. Rescaling also rounds, which adds up to half a unit, $0.005$.

Real noise is small against the scale. With $\Delta = 2^{50}$, an error of 8 units is $8 / 2^{50} \approx 7 \cdot 10^{-15}$. The figure is illustrative, but it shows why a scale of $2^{50}$ leaves plenty of room for scores. Noise matters when it gets close to the scale. At that point the low digits of the answer are garbage, and more levels or a bigger scale are the only fix.

!!! deep-dive "Why noise can only be set up in advance"
    The server cannot measure noise, because it sees only ciphertexts. So the parameters are chosen so that the whole computation, at its deepest, stays under the noise limit, and the client then checks accuracy on real data. If noise exceeded the limit, the client would still decrypt something and nothing would warn it.

## What CKKS cannot do

- **Exact answers.** Every result carries a small error. Two scores that differ by less than the error can come back in either order.
- **Comparisons.** There is no test for "greater than", "equal" or "in this set" on an encrypted number, and no `if`. Ombra works around this by returning encrypted scores to the client, which decrypts them and sorts. Filters are done by adding a penalty to the score of rows that do not match, so they drop out when the client ranks.
- **Unlimited depth.** The multiplication budget is fixed when the keys are generated: {{ preset("balanced", "multiplicative_depth") }} for both presets.
- **Speed.** Each multiplication touches every tower of every polynomial, which is why the server uses GPUs.
