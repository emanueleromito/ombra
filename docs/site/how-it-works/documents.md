# Documents and blobs

Every vector you add also produces encrypted blobs: one record blob that holds the exact vector and its metadata, and, if you pass a document, one document blob. The server stores them without being able to read them. This page covers how a blob is encrypted, why every blob has one of a few fixed sizes, and what happens to files too big to store inline.

The record blob is what makes [exact re-ranking](accuracy.md) possible. The search itself runs on the encrypted columns described under [encoding](encoding.md).

## Envelope encryption

Blobs use AES-256-GCM, an authenticated cipher: decryption fails if a single byte changed. Each blob gets its own fresh random 256-bit key, and that key is encrypted ("wrapped") under one long-lived key, the document master key. This is envelope encryption. The master key never encrypts bulk data directly, and compromising one blob key exposes one blob.

The blob keys are not derived from anything. Each one comes from the system random number generator when the blob is sealed. The master key is also random, 32 bytes, made when the client keys are created. It is stored together with the OpenFHE secret key in the key file, which is protected by a passphrase: the passphrase goes through scrypt (128 MiB of memory at the default cost, a random 16 byte salt) to give a key, and that key encrypts the master key and secret key with AES-256-GCM. The whole header, including the scrypt parameters and salt, is authenticated, so changing any of them makes opening fail. The server never sees the master key, the blob keys or the passphrase.

A blob is laid out like this (version 1, integers little-endian):

| Field | Bytes |
| --- | --- |
| magic `OMBD` | 4 |
| version | 2 |
| nonce for the key wrap | 12 |
| the wrapped blob key (32 byte key plus 16 byte tag) | 48 |
| nonce for the body | 12 |
| body: the padded plaintext encrypted with AES-256-GCM, with its 16 byte tag | the rest |

That is 78 bytes of header and a 16 byte body tag, so 94 bytes of overhead on every blob. Inside the encrypted body, the plaintext starts with a mode byte (text, bytes, reference or record) and the true length, then the payload, then zero padding.

Both encryptions also authenticate the blob's own identity: its kind (record or document), its collection name and its id. The body additionally authenticates the whole header. So a blob copied under another id or into another collection fails to open, even though the server could do that copy. The check does not stop the server from returning an older, valid version of the same blob. What the server can and cannot do is on [what the server learns](../security/what-the-server-learns.md).

## Size buckets

Blob length would leak how long a document is. To hide that, every stored blob has exactly one of a few sizes. The client pads the plaintext with zeros up to the smallest bucket that fits, and the server refuses a blob whose size is not a bucket size. The true length sits inside the ciphertext, so the server sees only the bucket.

The buckets are {{ const("ombra.blobstore.BUCKET_SIZES")[0] }}, {{ const("ombra.blobstore.BUCKET_SIZES")[1] }}, {{ const("ombra.blobstore.BUCKET_SIZES")[2] }} and {{ const("ombra.blobstore.BUCKET_SIZES")[3] }} bytes, which are 16 KiB, 256 KiB, 2 MiB and 16 MiB. A blob of $n$ plaintext bytes (payload plus the 9 byte inner header) goes into the smallest bucket $B$ with

$$n + 94 \le B .$$

For example, a record for a 768 dimension vector holds the vector as 768 float32 values, which is $768 \cdot 4 = 3072$ bytes. Add a 4 byte length field and, say, 60 bytes of JSON metadata, and the payload is 3136 bytes. With the 9 byte inner header, $n = 3145$, and $3145 + 94 = 3239 \le {{ const("ombra.blobstore.BUCKET_SIZES")[0] }}$, so the blob is the 16 KiB bucket and about 13 KB of it is padding. A 100,000 byte text document needs $100\,000 + 9 + 94 = 100\,103$ bytes, which does not fit in 16 KiB but fits in the 256 KiB bucket ({{ const("ombra.blobstore.BUCKET_SIZES")[1] }} bytes).

The cost is storage. Most records are well under one bucket, so each costs 16 KiB on the server whatever its content. For 100,000 vectors that is about 1.6 GB of record blobs, before documents. We have not measured blob storage on a real collection; the figure above is the bucket size times the count.

!!! deep-dive "What the bucket does and does not hide"
    Within one bucket the server cannot tell a 100 byte document from a 15,000 byte one. Across buckets it can: it learns that a document is above or below 16 KiB, 256 KiB and so on. The collection name, the blob kind and the id you chose are also visible. Ids are chosen by the caller, so do not use an id that carries meaning, such as a file name or an email address.

## References to external files

A large file does not belong in a blob. When a document body is bigger than the inline threshold (1 MiB by default) the blob stores a reference instead: a small encrypted JSON with the file's `uri`, its `sha256` and its `size`. The file itself stays in your own storage. A reference is a few hundred bytes, so it lands in the 16 KiB bucket.

When you read the document back, the client decrypts the reference, calls a fetch function you supply with the `uri`, and checks the result against the stored size and SHA-256. If either differs, it raises an error. So a file swapped in your storage is detected.

For example, a 5 MB PDF is above the 1 MiB threshold. The client stores `{"sha256": "...", "size": 5000000, "uri": "s3://bucket/report.pdf"}` as the document, and only that. A 600,000 byte text stays inline and lands in the 2 MiB bucket, since it does not fit in 256 KiB.

References are not wired into `Collection.add` yet. That call only seals inline documents, and it raises an error for a body over the inline threshold. To store a reference today, use `DocumentCodec.seal_reference` and `fetch_document` from the code below directly.

## What is not protected

The `uri` is encrypted inside the blob, but the fetch is yours: the request to your storage is not hidden by Ombra. Files in your storage are as protected as you make them. The document master key lives in the key file; lose the file and the passphrase and every record and document is unreadable. That is true of the secret key too, and there is no recovery.

The code is in [`client/ombra/documents.py`](https://github.com/erailabs/ombra/blob/main/client/ombra/documents.py) (seal and open), [`client/ombra/blobstore.py`](https://github.com/erailabs/ombra/blob/main/client/ombra/blobstore.py) (the bucket sizes and the store protocol) and [`client/ombra/keyfile.py`](https://github.com/erailabs/ombra/blob/main/client/ombra/keyfile.py) (the key file). The byte layout is also on the [file formats](../reference/file-formats.md) page.
