#!/usr/bin/env python3
"""Standard-library protocol and failure tests for the PC TCP client."""

import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLIENT_PATH = PROJECT_ROOT / "tools" / "edgevision_client.py"


def load_client_module():
    spec = importlib.util.spec_from_file_location("edgevision_client", CLIENT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CLIENT = load_client_module()


class OneShotServer:
    def __init__(self, response_chunks, expected_command=None, hold_open=0.0):
        self.response_chunks = response_chunks
        self.expected_command = expected_command
        self.hold_open = hold_open
        self.received = b""
        self.error = None
        self._listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._listener.bind(("127.0.0.1", 0))
        self._listener.listen(1)
        self.port = self._listener.getsockname()[1]
        self._thread = threading.Thread(target=self._serve, daemon=False)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._listener.close()
        self._thread.join(timeout=5.0)
        if self._thread.is_alive():
            raise AssertionError("mock server thread did not stop")
        if self.error is not None:
            raise self.error

    def _serve(self):
        try:
            connection, _ = self._listener.accept()
            with connection:
                connection.settimeout(5.0)
                chunks = []
                while b"\n" not in b"".join(chunks):
                    chunk = connection.recv(4096)
                    if not chunk:
                        break
                    chunks.append(chunk)
                self.received = b"".join(chunks)
                if self.expected_command is not None:
                    if self.received != self.expected_command:
                        raise AssertionError(
                            "unexpected command bytes: {!r}".format(self.received)
                        )
                for chunk in self.response_chunks:
                    connection.sendall(chunk)
                if self.hold_open > 0.0:
                    time.sleep(self.hold_open)
        except BaseException as error:  # surfaced by __exit__
            self.error = error


def run_client(*arguments, timeout=5.0):
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        [
            sys.executable,
            str(CLIENT_PATH),
            "--host",
            "127.0.0.1",
            "--port",
            str(arguments[0]),
            *arguments[1:],
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=environment,
        cwd=PROJECT_ROOT,
    )


class TcpClientProtocolTests(unittest.TestCase):
    def socket_pair(self):
        left, right = socket.socketpair()
        self.addCleanup(left.close)
        self.addCleanup(right.close)
        return left, right

    def test_fragmentation(self):
        client_socket, server_socket = self.socket_pair()
        server_socket.sendall(b'{"type":"eve')
        server_socket.sendall(b'nt","event":"ENTER","class":"person"')
        server_socket.sendall(b',"track_id":3,"confidence":0.9}\n')
        message = CLIENT.read_line(client_socket)
        self.assertEqual(message["type"], "event")
        self.assertEqual(message["event"], "ENTER")

    def test_coalescing(self):
        client_socket, server_socket = self.socket_pair()
        server_socket.sendall(
            b'{"type":"pong"}\n'
            b'{"type":"status","objects":1}\n'
            b'{"type":"event","event":"EXIT"}\n'
        )
        self.assertEqual(CLIENT.read_line(client_socket)["type"], "pong")
        self.assertEqual(CLIENT.read_line(client_socket)["type"], "status")
        self.assertEqual(CLIENT.read_line(client_socket)["type"], "event")

    def test_mixed_fragmentation_and_coalescing(self):
        client_socket, server_socket = self.socket_pair()
        server_socket.sendall(b'{"type":"st')
        server_socket.sendall(
            b'atus","objects":2}\n{"type":"eve'
            b'nt","event":"DWELL"}\n{"type":"pong"}\n'
        )
        self.assertEqual(CLIENT.read_line(client_socket)["type"], "status")
        self.assertEqual(CLIENT.read_line(client_socket)["type"], "event")
        self.assertEqual(CLIENT.read_line(client_socket)["type"], "pong")

    def test_clean_disconnect(self):
        client_socket, server_socket = self.socket_pair()
        server_socket.sendall(b'{"type":"pong"}\n')
        server_socket.close()
        self.assertEqual(CLIENT.read_line(client_socket)["type"], "pong")
        with self.assertRaisesRegex(RuntimeError, "connection closed"):
            CLIENT.read_line(client_socket)

    def test_partial_line_disconnect(self):
        client_socket, server_socket = self.socket_pair()
        server_socket.sendall(b'{"type":"status"')
        server_socket.close()
        with self.assertRaisesRegex(RuntimeError, "connection closed"):
            CLIENT.read_line(client_socket)

    def test_invalid_json(self):
        client_socket, server_socket = self.socket_pair()
        server_socket.sendall(b"this-is-not-json\n")
        with self.assertRaises(json.JSONDecodeError):
            CLIENT.read_line(client_socket)

        with OneShotServer([b"this-is-not-json\n"]) as server:
            result = run_client(server.port, "ping")
        self.assertEqual(result.returncode, 1)
        self.assertIn("malformed JSON response", result.stderr)

    def test_oversized_line(self):
        client_socket, server_socket = self.socket_pair()
        server_socket.sendall(b"x" * (CLIENT.MAX_RESPONSE_BYTES + 1))
        with self.assertRaisesRegex(RuntimeError, "exceeded 64 KiB"):
            CLIENT.read_line(client_socket)

    def test_cli_commands_and_newline_framing(self):
        cases = (
            ("ping", [b'{"type":"po', b'ng"}\n'], b"PING\n"),
            (
                "status",
                [b'{"type":"status","objects":0}\n'],
                b"GET_STATUS\n",
            ),
            (
                "subscribe",
                [
                    b'{"type":"status","subscribed":true}\n',
                    b'{"type":"event","event":"ENTER"}\n',
                ],
                b"SUBSCRIBE_EVENTS\n",
            ),
            (
                "unsubscribe",
                [b'{"type":"status","subscribed":false}\n'],
                b"UNSUBSCRIBE_EVENTS\n",
            ),
        )
        for command, response, expected in cases:
            with self.subTest(command=command):
                hold_open = 1.0 if command == "subscribe" else 0.0
                with OneShotServer(response, expected, hold_open) as server:
                    result = run_client(
                        server.port,
                        "--duration",
                        "0.1",
                        command,
                    )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(server.received, expected)

    def test_repeated_connections(self):
        for _ in range(50):
            with OneShotServer([b'{"type":"pong"}\n'], b"PING\n") as server:
                result = run_client(server.port, "ping")
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_short_localhost_soak(self):
        client_socket, server_socket = self.socket_pair()
        stop_at = time.monotonic() + 30.0

        def send_status_and_events():
            try:
                index = 0
                while time.monotonic() < stop_at:
                    if index % 5 == 0:
                        message = {"type": "event", "event": "ENTER", "track_id": index}
                    else:
                        message = {"type": "status", "objects": index % 4}
                    server_socket.sendall((json.dumps(message) + "\n").encode("utf-8"))
                    index += 1
                    time.sleep(0.01)
                server_socket.shutdown(socket.SHUT_WR)
            except OSError:
                pass

        sender = threading.Thread(target=send_status_and_events, daemon=False)
        sender.start()
        parsed = 0
        try:
            while sender.is_alive():
                try:
                    message = CLIENT.read_line(client_socket)
                except RuntimeError as error:
                    self.assertIn("connection closed", str(error))
                    break
                self.assertIn(message["type"], ("status", "event"))
                parsed += 1
        finally:
            sender.join(timeout=5.0)
        self.assertFalse(sender.is_alive())
        self.assertGreaterEqual(parsed, 1000)


if __name__ == "__main__":
    unittest.main()
