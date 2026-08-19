# PC event logger

The lightweight Windows event logger subscribes to the existing RK3568 TCP
event stream and writes one CSV file for manual verification.

```text
edgevision_event_logger.exe
```

Defaults:

- board: `192.168.77.2`
- port: `9000`
- output: `event_log.csv`

Optional arguments:

```text
edgevision_event_logger.exe --host 192.168.77.2 --port 9000 --output event_log.csv
```

The logger sends `SUBSCRIBE_EVENTS`, ignores the initial status response, and
stores only:

```text
timestamp,event,logical_id,class
```

It accepts `ENTER`, `DWELL`, and `EXIT`. Press Ctrl+C to stop and close the CSV.
