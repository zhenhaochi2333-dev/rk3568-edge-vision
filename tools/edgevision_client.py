#!/usr/bin/env python3
"""Small standard-library client for the EdgeVision Phase 4 TCP protocol."""

import argparse
import json
import socket
import sys
import time

DEFAULT_HOST = "192.168.77.2"
MAX_RESPONSE_BYTES = 64 * 1024


def read_line(connection):
    data = bytearray()
    while True:
        chunk = connection.recv(1)
        if not chunk:
            raise RuntimeError("EdgeVision TCP connection closed")
        data.extend(chunk)
        if len(data) > MAX_RESPONSE_BYTES:
            raise RuntimeError("EdgeVision TCP response exceeded 64 KiB")
        if chunk == b"\n":
            return json.loads(data.decode("utf-8"))


def send_command(connection, command):
    connection.sendall((command + "\n").encode("ascii"))
    return read_line(connection)


def format_event(event):
    return (
        "[{timestamp}] {event} {class_name} #{track_id} confidence={confidence:.3f}"
        .format(
            timestamp=time.strftime("%H:%M:%S"),
            event=event.get("event", "EVENT"),
            class_name=event.get("class", "unknown"),
            track_id=event.get("track_id", -1),
            confidence=event.get("confidence", 0.0),
        )
    )


def main():
    parser = argparse.ArgumentParser(description="EdgeVision TCP status/event client")
    parser.add_argument("--host", default=DEFAULT_HOST,
                        help="RK3568 address (default: %(default)s)")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("command", choices=("ping", "status", "subscribe", "unsubscribe"))
    parser.add_argument("--duration", type=float, default=10.0,
                        help="seconds to listen after subscribe")
    args = parser.parse_args()

    command_map = {
        "ping": "PING",
        "status": "GET_STATUS",
        "subscribe": "SUBSCRIBE_EVENTS",
        "unsubscribe": "UNSUBSCRIBE_EVENTS",
    }
    try:
        with socket.create_connection((args.host, args.port), timeout=5.0) as connection:
            connection.settimeout(1.0)
            response = send_command(connection, command_map[args.command])
            print(json.dumps(response, ensure_ascii=False))
            if args.command != "subscribe":
                return 0

            deadline = time.monotonic() + max(0.0, args.duration)
            while time.monotonic() < deadline:
                try:
                    event = read_line(connection)
                except socket.timeout:
                    continue
                if event.get("type") == "event":
                    print(format_event(event))
                else:
                    print(json.dumps(event, ensure_ascii=False))
        return 0
    except KeyboardInterrupt:
        print("\nedgevision_client: interrupted", file=sys.stderr)
        return 130
    except (OSError, RuntimeError, json.JSONDecodeError) as error:
        print("edgevision_client: {}".format(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
