# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
"""Back up and check key files.

    ombra-keys backup KEYFILE DESTINATION [--force] [--check-passphrase]
    ombra-keys check KEYFILE [--tag TAG | --server HOST:PORT --collection NAME]

`python -m ombra.keys` does the same. The passphrase is never an argument (it would show in the
process list): it comes from a prompt, from an environment variable (`--passphrase-env NAME`, or
OMBRA_KEY_PASSPHRASE if that is set), from a file (`--passphrase-file`) or from an open file
descriptor (`--passphrase-fd`). `check` opens the file, checks that its secret and public key work
and carry one key tag, and compares that tag with the one you pass (`--tag`, offline) or with the
tag of a collection on a server (`--server`, `--collection`; the API key is read from the
environment variable OMBRA_API_KEY, `--insecure` is for a loopback server without TLS). It prints
OK or the reason it failed. It never prints a key, the passphrase or any other secret.
Exit code: 0 OK, 1 a check failed, 2 bad usage.
"""

from __future__ import annotations

import argparse
import getpass
import os
import sys
from collections.abc import Sequence

from ombra.keyfile import KeyFileError, backup_key_file, check_key_file

PASSPHRASE_ENV = "OMBRA_KEY_PASSPHRASE"  # noqa: S105 - the name of a variable, not a secret
API_KEY_ENV = "OMBRA_API_KEY"


def _passphrase_arguments(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("passphrase (never an argument)")
    g.add_argument("--passphrase-env", metavar="NAME", help="environment variable that holds it")
    g.add_argument("--passphrase-file", metavar="PATH", help="file whose first line is it")
    g.add_argument("--passphrase-fd", type=int, metavar="N", help="open descriptor to read it from")


def _read_passphrase(args: argparse.Namespace, *, ask: bool) -> str | None:
    """The passphrase from the chosen source. With none chosen: PASSPHRASE_ENV if set, else a
    prompt when `ask`, else None."""
    if args.passphrase_fd is not None:
        with os.fdopen(args.passphrase_fd, "r", closefd=False) as f:
            value = f.readline().rstrip("\r\n")
    elif args.passphrase_file:
        with open(args.passphrase_file, encoding="utf-8") as f:
            value = f.readline().rstrip("\r\n")
    elif args.passphrase_env:
        value = os.environ.get(args.passphrase_env, "")
        if not value:
            raise KeyFileError("the passphrase environment variable is not set")
    elif os.environ.get(PASSPHRASE_ENV):
        value = os.environ[PASSPHRASE_ENV]
    elif ask:
        value = getpass.getpass("Key file passphrase: ")
    else:
        return None
    if not value:
        raise KeyFileError("the passphrase is empty")
    return value


def _server_tag(address: str, collection: str, insecure: bool) -> str:
    """The key tag of `collection` on the server (GetCollection; needs a read key)."""
    from ombra._transport import GrpcTransport
    from ombra.errors import OmbraError

    api_key = None if insecure else (os.environ.get(API_KEY_ENV) or None)
    tr = GrpcTransport(address, api_key, None, insecure)
    try:
        r = tr.stub.GetCollection(tr.pb.GetCollectionRequest(name=collection))
    except OmbraError as exc:
        raise KeyFileError(f"could not ask the server: {exc}") from None
    finally:
        tr.close()
    if r.WhichOneof("result") == "error":
        raise KeyFileError("the server does not have that collection")
    return str(r.info.spec.key_tag)


def _cmd_backup(args: argparse.Namespace) -> int:
    pw = _read_passphrase(args, ask=args.check_passphrase)
    backup_key_file(args.keyfile, args.destination, passphrase=pw, force=args.force)
    what = "structure and passphrase" if pw is not None else "structure"
    print(f"OK: backup written and checked ({what})")
    if pw is None:
        print("note: pass a passphrase source or --check-passphrase to also open the copy")
    return 0


def _cmd_check(args: argparse.Namespace) -> int:
    pw = _read_passphrase(args, ask=True)
    assert pw is not None
    res = check_key_file(args.keyfile, pw)
    expected, where = args.tag, "the tag you gave"
    if args.server is not None:
        expected, where = _server_tag(args.server, args.collection, args.insecure), "the server"
    if expected is not None and expected != res.key_tag:
        print(f"FAIL: the key file's key tag does not match {where}", file=sys.stderr)
        return 1
    print(f"OK: key file opens, secret and public key match, key tag {res.key_tag}")
    if expected is not None:
        print(f"OK: key tag matches {where}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="ombra-keys", description="Back up and check key files.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("backup", help="copy a key file atomically and check the copy")
    b.add_argument("keyfile")
    b.add_argument("destination")
    b.add_argument("--force", action="store_true", help="replace an existing destination")
    b.add_argument(
        "--check-passphrase", action="store_true", help="ask for the passphrase, open the copy"
    )
    _passphrase_arguments(b)
    c = sub.add_parser("check", help="open a key file and compare its key tag")
    c.add_argument("keyfile")
    c.add_argument("--tag", help="the key tag to compare with (offline)")
    c.add_argument("--server", metavar="HOST:PORT", help="compare with a collection on a server")
    c.add_argument("--collection", help="the collection name on the server")
    c.add_argument("--insecure", action="store_true", help="no TLS (loopback test servers)")
    _passphrase_arguments(c)
    args = ap.parse_args(argv)
    if args.cmd == "check":
        if args.tag is not None and args.server is not None:
            ap.error("use --tag or --server, not both")
        if (args.server is None) != (args.collection is None):
            ap.error("--server and --collection go together")
    try:
        return _cmd_backup(args) if args.cmd == "backup" else _cmd_check(args)
    except (KeyFileError, OSError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
