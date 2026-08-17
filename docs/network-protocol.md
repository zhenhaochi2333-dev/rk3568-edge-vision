# EdgeVision Phase 4 TCP Protocol

This document records the TCP protocol currently implemented by the Linux
`edge_vision` target. It is a status/event transport only.

## Endpoint

- Server: RK3568
- Default address used by the PC client: `192.168.77.2`
- Default port: `9000`
- Transport: one TCP connection with one active client at a time

When a new client connects, the server closes the previous active client.

## Framing and limits

Commands are newline-delimited ASCII lines. The maximum command body is 1024
bytes, excluding the terminating newline. Responses are newline-delimited
NDJSON: one JSON object per line.

The PC client imposes a separate 64 KiB maximum on one response line. This is
a client-side safety limit; it does not change the server's 1024-byte command
limit.

The server keeps a bounded event queue of 64 entries. If it becomes full, the
oldest queued event is dropped. Events are produced from the formal
`RegionMonitor` event object; the TCP layer does not synthesize ROI events.

## Commands

### `PING`

Request:

```text
PING
```

Response:

```json
{"type":"pong"}
```

### `GET_STATUS`

Request:

```text
GET_STATUS
```

Response:

```json
{"type":"status","objects":2,"camera_fps":30.000,"display_fps":15.000,"detection_fps":7.500,"uptime_ms":1234,"subscribed":false}
```

`uptime_ms` is server uptime. `camera_fps`, `display_fps`, and
`detection_fps` are the latest values published by the application.

### `SUBSCRIBE_EVENTS`

Request:

```text
SUBSCRIBE_EVENTS
```

The first response is a status object with `subscribed:true`. Subsequent
events are sent as NDJSON lines while the client remains connected.

Example event:

```json
{"type":"event","event":"ENTER","class":"person","track_id":7,"confidence":0.910,"timestamp_ms":123456789}
```

Supported event names are `ENTER`, `DWELL`, and `EXIT`. `timestamp_ms` is the
source monotonic timestamp in milliseconds, not wall-clock UTC.

### `UNSUBSCRIBE_EVENTS`

Request:

```text
UNSUBSCRIBE_EVENTS
```

Response is a status object with `subscribed:false`; queued events for that
subscription are cleared.

## Errors

Unknown or overlong commands return an NDJSON error object, for example:

```json
{"type":"error","message":"unknown_command"}
```

The PC client reports connection closure, socket errors, malformed JSON, and
Ctrl+C as clear local errors/statuses. Running the client again creates a new
TCP connection, so reconnecting does not require persistent client state.
