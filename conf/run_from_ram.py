#!/usr/bin/env python3
"""Exec a binary from an in-memory copy (memfd) instead of its file on disk.

Under heavy ZFS load the kernel evicted mediamtx's code pages, and faulting them back in
from the docker zvol froze every stream for seconds at a time (threads stuck in
filemap_fault). A memfd copy lives in RAM, so its pages are never re-read from disk.
The container's tmpfs mounts are noexec, which rules out copying the binary to one.

Usage: run_from_ram.py <binary> [args...]
"""
import os
import sys

binary, args = sys.argv[1], sys.argv[1:]
name = os.path.basename(binary)
fd = os.memfd_create(name, 0)
with open(binary, "rb") as f:
    data = memoryview(f.read())
while data:
    data = data[os.write(fd, data):]
os.set_inheritable(fd, True)
os.execv(f"/proc/self/fd/{fd}", [name] + args[1:])
