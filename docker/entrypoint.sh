#!/usr/bin/env bash
# Start a virtual screen and serve it to the browser at
# http://localhost:6080/vnc.html, then run the given command.
set -e
Xvfb :1 -screen 0 1600x900x24 >/tmp/xvfb.log 2>&1 &
sleep 1
fluxbox >/tmp/fluxbox.log 2>&1 &
x11vnc -display :1 -forever -shared -nopw -rfbport 5900 -quiet >/tmp/x11vnc.log 2>&1 &
websockify --web /usr/share/novnc 6080 localhost:5900 >/tmp/novnc.log 2>&1 &
exec "$@"
