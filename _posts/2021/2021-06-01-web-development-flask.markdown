---
layout: post
title: Web Development - Flask
date: '2021-06-01 13:19'
subtitle: React, Flask
comments: true
tags:
    - Computer Vision
---

## Jinja

Jinja is a templating language, not a general-purpose programming language. It lives on the **server inside** Flask (or other Python frameworks) and provides you with:

- Template syntax `({{ … }}` for expressions, `{% … %}` for control structures like loops/ifs.
- Filters and tests to transform or inspect data (e.g. {{ user.name|upper }}, {% if items|length > 0 %}).

When you call Flask’s `render_template("page.html", foo=bar)`, Jinja runs on the server, stitches your Python data into the template, and emits plain HTML (with CSS, etc.) to the browser.

Where does JavaScript come in?

- Server-side rendering (Jinja)
    - All the dynamic bits you write in `{{ … }} or {% … %}` are resolved before the browser ever sees the page. The client gets a finalized HTML document.

- Client-side dynamics (JavaScript)
    - If you want interactive behavior after the page loads—DOM updates on clicks, **real-time data without full reloads, animations, form validation**—you write JavaScript (vanilla JS or a framework like React/Vue). That JS runs in the browser and can fetch new data (via fetch, WebSockets, SSE, etc.) and manipulate the DOM.

Quick Example:

```
<!-- templates/hello.html -->
<!doctype html>
<html>
  <head><title>Hello</title></head>
  <body>
    <h1>Hello, {{ user_name }}!</h1>          {# Jinja: filled in on the server #}
    <button id="btn">Click me</button>
    <p id="msg"></p>

    <script>
      // JavaScript: runs in the browser after the HTML arrives
      document.getElementById('btn').onclick = () => {
        document.getElementById('msg').textContent = 'You clicked!';
      };
    </script>
  </body>
</html>
```

- The `{{ user_name }}` bit is Jinja, resolved by Flask before shipping the HTML.
- The `<script>` block is pure JavaScript, responsible for client-side interactivity after the page loads.