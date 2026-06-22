#!/usr/bin/env python3
"""Check which IP/port the PICO headset can reach on this PC.

Run this, then enter the shown IP on the headset. Any successful connect prints here.

  python scripts/visualization/vr_stream_diagnose.py
"""

from __future__ import annotations

import socket
import sys
import threading

PORTS = (12345, 63901, 8080, 9000)


def get_network_interfaces():
    import re
    import subprocess

    entries = []
    try:
        out = subprocess.check_output(["ip", "-4", "-o", "addr", "show"], text=True)
        for line in out.splitlines():
            m = re.match(r"\d+:\s+(\S+)\s+inet\s+([\d.]+)/", line)
            if m and not m.group(2).startswith("127."):
                entries.append((m.group(1), m.group(2)))
    except (OSError, subprocess.CalledProcessError):
        pass
    return entries


def listen_port(port: int) -> None:
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        srv.bind(("0.0.0.0", port))
    except OSError as exc:
        print(f"[port {port}] cannot bind: {exc}")
        return
    srv.listen(5)
    print(f"[port {port}] listening")
    while True:
        conn, peer = srv.accept()
        print(f"[port {port}] *** CONNECTED from {peer[0]}:{peer[1]} ***")
        conn.close()


def main() -> int:
    print("PC IPv4 addresses (use the one on the same LAN as PICO):")
    for iface, ip in get_network_interfaces():
        tag = " ← Wi‑Fi" if iface.startswith("wl") else ""
        print(f"  {ip}  ({iface}){tag}")
    print(f"\nWatching ports {PORTS}. On PICO, try each IP above with Listen.\n")

    threads = []
    for port in PORTS:
        t = threading.Thread(target=listen_port, args=(port,), daemon=True)
        t.start()
        threads.append(t)

    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        print("\nStopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
