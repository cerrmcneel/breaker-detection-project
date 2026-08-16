#!/usr/bin/env python3
"""Keep raw_uploads identical on the training machine and the production VM.

Why this exists
---------------
Photos submitted through panelsafe.cv land on VM 101 and nowhere else, while
annotation and training happen on the workstation. On 2026-08-11 the two had
silently diverged in BOTH directions -- the VM held 5 real uploads the trainer
had never seen, and the trainer held images the VM did not. That is two problems
at once: the uploads are unbacked-up, and a retrain would quietly run on stale
data while believing it had everything.

Behaviour
---------
* **Additive only. Nothing is ever deleted**, on either side.
* Files are matched by CONTENT (sha256 of the bytes), not by name, so an image
  already present under a different filename is not copied in again.
* Same name + different content is reported as a CONFLICT and skipped, never
  overwritten. Resolve those by hand.
* ``--dry-run`` reports exactly what would move without touching anything.

Usage
-----
    python -m src.tools.sync_uploads --dry-run
    python -m src.tools.sync_uploads
    python -m src.tools.sync_uploads --pull-only     # VM -> workstation (backup)

Environment (optional; defaults target the current deployment)
    PANELSAFE_VM_HOST        e.g. eric-mcneel@192.168.1.168
    PANELSAFE_VM_UPLOAD_DIR  path to raw_uploads on the VM
"""
import argparse
import hashlib
import os
import subprocess
import sys
import tempfile

LOCAL_DIR = os.path.join("data", "images", "raw_uploads")
VM_HOST = os.getenv("PANELSAFE_VM_HOST", "eric-mcneel@192.168.1.168")
VM_DIR = os.getenv("PANELSAFE_VM_UPLOAD_DIR",
                   "~/breaker-detection-project/data/images/raw_uploads")

VM_CONTAINER = os.getenv("PANELSAFE_VM_CONTAINER", "breaker-app")
CONTAINER_UPLOAD_DIR = os.getenv("PANELSAFE_CONTAINER_UPLOAD_DIR",
                                 "/code/data/images/raw_uploads")

SYNCED_EXTS = (".jpg", ".jpeg", ".png", ".gif", ".txt")
SSH_OPTS = ["-o", "ConnectTimeout=20", "-o", "BatchMode=yes"]

REMOTE_INVENTORY = r"""
import hashlib, os, sys
d = os.path.expanduser(sys.argv[1])
exts = ('.jpg', '.jpeg', '.png', '.gif', '.txt')
for fn in sorted(os.listdir(d)) if os.path.isdir(d) else []:
    p = os.path.join(d, fn)
    if not os.path.isfile(p) or not fn.lower().endswith(exts):
        continue
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    print(h.hexdigest(), fn, sep='\t')
"""


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def local_inventory():
    out = {}
    if not os.path.isdir(LOCAL_DIR):
        return out
    for fn in sorted(os.listdir(LOCAL_DIR)):
        p = os.path.join(LOCAL_DIR, fn)
        if os.path.isfile(p) and fn.lower().endswith(SYNCED_EXTS):
            out[fn] = sha256_file(p)
    return out


def remote_inventory():
    # The script goes over stdin, not as `python3 -c`: ssh re-parses its argument
    # list through the remote shell, which mangles a multi-line program. With
    # `python3 - <dir>` the remote shell still expands the leading ~ in VM_DIR.
    proc = subprocess.run(
        ["ssh", *SSH_OPTS, VM_HOST, f"python3 - {VM_DIR}"],
        input=REMOTE_INVENTORY, capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise SystemExit(f"could not inventory {VM_HOST}:{VM_DIR}\n{proc.stderr.strip()}")
    out = {}
    for line in proc.stdout.splitlines():
        if "\t" in line:
            digest, name = line.split("\t", 1)
            out[name] = digest
    return out


def _filelist(names):
    """tar -T needs a real file; names may contain spaces, one per line."""
    fh = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False,
                                     encoding="utf-8", newline="\n")
    fh.write("\n".join(names) + "\n")
    fh.close()
    return fh.name


def pull(names):
    """VM -> local, streamed as one tar so it is a single SSH round trip."""
    listfile = _filelist(names)
    try:
        with open(listfile, "rb") as stdin:
            tar = subprocess.run(
                ["ssh", *SSH_OPTS, VM_HOST, f"cd {VM_DIR} && tar czf - -T -"],
                stdin=stdin, capture_output=True,
            )
        if tar.returncode != 0:
            raise SystemExit(f"pull failed: {tar.stderr.decode(errors='replace')[:400]}")
        extract = subprocess.run(["tar", "xzf", "-", "-C", LOCAL_DIR],
                                 input=tar.stdout, capture_output=True)
        if extract.returncode != 0:
            raise SystemExit(f"extract failed: {extract.stderr.decode(errors='replace')[:400]}")
    finally:
        os.unlink(listfile)


def remote_dir_is_writable():
    """Can the SSH user write into raw_uploads directly?

    Usually NOT: the gateway container runs as root and creates that directory
    root-owned 0755, so the login user gets permission denied. Probing beats
    assuming either way, since a chown would silently change the answer.
    """
    probe = subprocess.run(
        ["ssh", *SSH_OPTS, VM_HOST, f"test -w {VM_DIR} && echo yes || echo no"],
        capture_output=True, text=True,
    )
    return probe.stdout.strip() == "yes"


def push(names, via_container=None):
    """local -> VM as one tar stream.

    When raw_uploads is root-owned, the archive is piped into the gateway
    container instead, which runs as root over the same bind mount. That needs
    only docker-group membership -- no sudo, no chown, no ownership change to the
    files the app itself writes.
    """
    listfile = _filelist(names)
    try:
        tar = subprocess.run(["tar", "czf", "-", "-C", LOCAL_DIR, "-T", listfile],
                             capture_output=True)
        if tar.returncode != 0:
            raise SystemExit(f"archive failed: {tar.stderr.decode(errors='replace')[:400]}")

        if via_container:
            remote_cmd = (f"docker exec -i {via_container} "
                          f"tar xzf - -C {CONTAINER_UPLOAD_DIR}")
        else:
            remote_cmd = f"mkdir -p {VM_DIR} && tar xzf - -C {VM_DIR}"

        send = subprocess.run(["ssh", *SSH_OPTS, VM_HOST, remote_cmd],
                              input=tar.stdout, capture_output=True)
        if send.returncode != 0:
            raise SystemExit(f"push failed: {send.stderr.decode(errors='replace')[:400]}")
    finally:
        os.unlink(listfile)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="report without transferring")
    ap.add_argument("--pull-only", action="store_true", help="VM -> workstation only")
    ap.add_argument("--push-only", action="store_true", help="workstation -> VM only")
    args = ap.parse_args()

    os.makedirs(LOCAL_DIR, exist_ok=True)
    local, remote = local_inventory(), remote_inventory()
    local_hashes, remote_hashes = set(local.values()), set(remote.values())

    # Match on content, so a file already present under another name is not re-copied.
    to_pull = sorted(n for n, h in remote.items() if h not in local_hashes)
    to_push = sorted(n for n, h in local.items() if h not in remote_hashes)
    conflicts = sorted(n for n in set(local) & set(remote) if local[n] != remote[n])

    print(f"local : {len(local):>4} files ({len(local_hashes)} distinct)")
    print(f"VM    : {len(remote):>4} files ({len(remote_hashes)} distinct)")
    print(f"\nmissing locally (would PULL): {len(to_pull)}")
    for n in to_pull:
        print(f"  <- {n}")
    print(f"\nmissing on VM (would PUSH):   {len(to_push)}")
    for n in to_push[:40]:
        print(f"  -> {n}")
    if len(to_push) > 40:
        print(f"  ... and {len(to_push) - 40} more")

    if conflicts:
        print(f"\n!! {len(conflicts)} CONFLICT(S): same name, different content -- skipped, resolve by hand")
        for n in conflicts:
            print(f"  ?? {n}")

    if args.dry_run:
        print("\ndry run: nothing transferred")
        return 0

    if to_pull and not args.push_only:
        pull(to_pull)
        print(f"\npulled {len(to_pull)} file(s) from the VM")
    if to_push and not args.pull_only:
        via = None if remote_dir_is_writable() else VM_CONTAINER
        if via:
            print(f"\n{VM_DIR} is not writable by the login user (container-owned); "
                  f"pushing via `docker exec {via}` instead")
        push(to_push, via_container=via)
        print(f"pushed {len(to_push)} file(s) to the VM")
    if not to_pull and not to_push:
        print("\nalready in sync")

    # Re-inventory so the exit state is verified, not assumed.
    local2, remote2 = local_inventory(), remote_inventory()
    only_local = set(local2.values()) - set(remote2.values())
    only_remote = set(remote2.values()) - set(local2.values())
    print(f"\nafter sync: local {len(local2)} / VM {len(remote2)}  "
          f"| content only-local {len(only_local)}, only-VM {len(only_remote)}")
    if not args.pull_only and not args.push_only and (only_local or only_remote):
        print("WARNING: still divergent after a full sync -- check the conflicts above")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
