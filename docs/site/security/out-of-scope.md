# Out of scope

This page lists what Ombra does not protect against. Read it before you put sensitive data in.

## A server that lies

The threat model is an honest-but-curious server. A malicious server can do any of the following, and nothing in the client will notice:

- Return wrong or made-up scores. Scores are ciphertexts without an authenticity tag, so the client decrypts whatever it receives.
- Leave clusters out of a search, or return fewer results than exist.
- Return an older version of a record or document blob. Each blob is authenticated against its collection, kind and id, so a blob moved to another id fails to open, but the server can serve a stale copy of the right id.
- Refuse to answer, or delete data. Back up your data, see [Backups](../operations/backups.md).

The final ranking is exact because the client re-scores the top candidates with the real vectors from the blob store. That corrects the noise of CKKS scores. It does not stop a server from hiding the best match.

## Access patterns and metadata

Encryption hides values, not behavior. The server sees which ids you insert, which clusters you ask for, which record blobs you fetch and when. The full list is in [What the server learns](what-the-server-learns.md). In short:

- Vector ids are plaintext. Use ids that carry no meaning, such as random identifiers, not email addresses or file names.
- Sizes: the number of vectors, the number of clusters, the vectors per cluster, the filter field count and values per field, and the size class of each blob.
- The number and timing of queries, and the ids of the candidate records the client fetches after each search.

Strict mode removes the cluster choice from this list. It does not remove the blob fetches.

## Side channels

We have not tested for or defended against timing, cache, power or memory-access side channels on the server or the client. The GPU kernels are not written to be constant time. Anyone who can run code next to the secret key on the client, or measure the client's decryption, is outside the model. The server sees how long each round takes, and in Fast and Decoy that time depends on which clusters were asked.

## Repeated queries

Decoys hide the real clusters of one query. A server that watches many queries can compare them: the real clusters of related queries repeat while the random decoys change, so over time the real ones stand out. We have not measured how fast. Strict is the only level without this weakness, at the cost of scanning every cluster.

## Compromised clients and keys

- Whoever has the key file and passphrase can decrypt everything. There is no per-user access control inside a collection: every client that holds the key reads every vector and document.
- Losing the key file means losing the data. Nothing on the server can recover it.
- An API key is an access token, not an encryption key. It limits who can send requests and what they can do. Encryption protects the data.
- Local mode runs the server code in a child process on your machine. There is no trust boundary between the two, so it is for trying Ombra out, not for hiding data from the machine it runs on.

## Not analysed or not built

- Attack models beyond 128-bit classical security (see [Parameters and security level](parameters.md)). We make no post-quantum claim.
- Differential privacy, oblivious access, private information retrieval and verifiable computation. None is implemented.
- Protection of the embedding model. The vectors come from your model and can reveal what it knows about the text. Ombra encrypts them. It does not make them harmless once decrypted.
- Anonymity of the client. The server sees the client's network address and API key.
