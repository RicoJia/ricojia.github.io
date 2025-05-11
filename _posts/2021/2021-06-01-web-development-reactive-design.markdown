---
layout: post
title: Web Development - Reactive Design
date: '2021-06-01 13:19'
subtitle: React, Flask
comments: true
tags:
    - Computer Vision
---

## React

React itself is just a UI library and doesn’t bundle any network protocol or “live-update” mechanism under the hood. It re-renders your components whenever you call setState (or update state via Hooks), but it doesn’t open sockets or push data by default.

### How React updates the UI

1. You call `setState` (class) or a state-updater from useState/useReducer.
2. React schedules a render, `diffs the virtual DOM`, and patches the real DOM. No network I/O is involved in this process—it’s pure JavaScript in the browser.

If you want server-driven “push” updates, like GraphQL subscriptions, etc., you must explicitly open a WebSocket (or Server-Sent Events) yourself and then hook its messages into your React state. For example:

```javascript
import { useState, useEffect } from 'react';

function LiveDataComponent() {
  const [data, setData] = useState(null);

  useEffect(() => {
    const ws = new WebSocket('wss://example.com/updates');
    ws.onmessage = evt => {
      const parsed = JSON.parse(evt.data);
      setData(parsed);
    };
    return () => ws.close();
  }, []);

  if (!data) return <div>Loading…</div>;
  return <pre>{JSON.stringify(data, null, 2)}</pre>;
}
```

## "Reactiveness" In Flask

 Flask is a server-side Python web framework, so it **doesn’t have React’s concept of component-level state Hooks** like `useState` or `useEffect`. What Flask does provide are request-lifecycle hooks—decorators you can register to run code at various points in handling an HTTP request:

    ```python
    @app.before_request

    @app.after_request

    @app.teardown_request

    @app.before_first_request
    ```
These let you run setup or teardown logic around each incoming request, but they have nothing to do with client-side reactive state.

To reactively update a website:

- Polling: Each poll is a full HTTP request/response (handshake, headers, routing)
- WebSockets (long-lived connection where server pushes updates). You have one persistent connection that comes with a lower per-message overhead
- Server-Sent-Events: in a react front end + flask backend architecture, 
    - we set up an event stream that constantly pushes on the flast side:
        ```python
        def sse_stream():
            cnt = 0
            while True:
                time.sleep(1)
                cnt += 1
                yield f"data: tick {cnt}\n\n"
        @app.route("/api/stream")
        def stream():
            return Response(
                sse_stream(),
                mimetype="text/event-stream",
                headers={"Cache-Control": "no-cache"}
            )
        ```
        - Bring it app using `python app.py`
    - And we set up a react front-end with a proxy:
        ```
        function App() {
            const [rest, setRest] = useState(null);
            const [sse,  setSse]  = useState("");
            const [ws,   setWs]   = useState("");

            // REST polling (once)
            useEffect(() => {
                fetch("/api/data")
                .then(r=>r.json())
                .then(j=>setRest(j.value));
            }, []);

            // SSE
            useEffect(() => {
                const es = new EventSource("/api/stream");
                es.onmessage = e => setSse(e.data);
                return () => es.close();
            }, []);
        }
        ```
        - and bring it up by `npm start`
