"""
IoT Dashboard Server

Receives POST /scan requests from the CV pipeline and serves a live
dashboard showing the latest scan result and a history log.

Run with:
    python server/app.py

Then open http://127.0.0.1:5000 in a browser.
The dashboard auto-refreshes every 2 seconds.
"""

from collections import deque
from datetime import datetime

from flask import Flask, jsonify, render_template_string, request

app = Flask(__name__)

# In-memory log — keeps the last 50 scans
_scan_log  = deque(maxlen=50)
_latest    = None

# ---------------------------------------------------------------------------
# Dashboard HTML (inline so no separate templates folder is needed)
# ---------------------------------------------------------------------------
_DASHBOARD = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta http-equiv="refresh" content="2">
  <title>Disability Card Validator</title>
  <style>
    body { font-family: Arial, sans-serif; background: #1a1a2e; color: #eee; margin: 0; padding: 24px; }
    h1   { color: #e0e0e0; margin-bottom: 4px; }
    .sub { color: #888; margin-bottom: 32px; font-size: 0.9em; }

    .result-box {
      display: inline-block;
      padding: 20px 40px;
      border-radius: 12px;
      font-size: 2.4em;
      font-weight: bold;
      margin-bottom: 24px;
      letter-spacing: 2px;
    }
    .valid   { background: #1b5e20; color: #a5d6a7; border: 2px solid #66bb6a; }
    .invalid { background: #b71c1c; color: #ef9a9a; border: 2px solid #ef5350; }
    .none    { background: #333; color: #aaa; border: 2px solid #555; }

    .meta { color: #aaa; font-size: 0.9em; margin-bottom: 32px; }
    .meta span { margin-right: 20px; }

    table { width: 100%; border-collapse: collapse; font-size: 0.88em; }
    th    { text-align: left; padding: 8px 12px; background: #16213e; color: #90caf9; border-bottom: 1px solid #333; }
    td    { padding: 7px 12px; border-bottom: 1px solid #222; }
    tr.v  { background: #1b3a1f; }
    tr.i  { background: #3a1a1a; }
  </style>
</head>
<body>
  <h1>Disability Card Validator</h1>
  <p class="sub">Auto-refreshes every 2 seconds &nbsp;|&nbsp; Last updated: {{ now }}</p>

  {% if latest %}
    <div class="result-box {{ 'valid' if latest.is_valid else 'invalid' }}">
      {{ 'VALID' if latest.is_valid else 'INVALID' }}
    </div>
    <div class="meta">
      <span>Card: <strong>{{ latest.card_type or 'unknown' }}</strong></span>
      <span>Score: <strong>{{ '%.2f' % latest.score }}</strong></span>
      <span>Colour: {{ '%.2f' % latest.colour_conf }}</span>
      <span>Text: {{ '%.2f' % latest.text_conf }}</span>
      <span>Layout: {{ '%.2f' % latest.layout_conf }}</span>
      <span>ML: {{ '%.2f' % latest.ml_conf }}</span>
      <span>Attempts: <strong>{{ latest.attempts or '—' }}</strong></span>
      <br><small style="color:#666">{{ latest.timestamp }}</small>
    </div>
  {% else %}
    <div class="result-box none">WAITING FOR SCAN</div>
  {% endif %}

  <h2 style="color:#90caf9; margin-top: 32px;">Scan History</h2>
  {% if log %}
  <table>
    <tr>
      <th>Time</th><th>Result</th><th>Card Type</th>
      <th>Score</th><th>Colour</th><th>Text</th><th>Layout</th><th>ML</th><th>Attempts</th>
    </tr>
    {% for s in log %}
    <tr class="{{ 'v' if s.is_valid else 'i' }}">
      <td>{{ s.timestamp }}</td>
      <td><strong>{{ 'VALID' if s.is_valid else 'INVALID' }}</strong></td>
      <td>{{ s.card_type or '—' }}</td>
      <td>{{ '%.2f' % s.score }}</td>
      <td>{{ '%.2f' % s.colour_conf }}</td>
      <td>{{ '%.2f' % s.text_conf }}</td>
      <td>{{ '%.2f' % s.layout_conf }}</td>
      <td>{{ '%.2f' % s.ml_conf }}</td>
      <td>{{ s.attempts or '—' }}</td>
    </tr>
    {% endfor %}
  </table>
  {% else %}
    <p style="color:#666">No scans yet.</p>
  {% endif %}
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.post("/scan")
def receive_scan():
    """Receive a scan result from the CV pipeline."""
    global _latest
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "no JSON body"}), 400

    _latest = data
    _scan_log.appendleft(data)  # newest first
    return jsonify({"status": "ok"}), 200


@app.get("/")
def dashboard():
    """Serve the live dashboard."""
    return render_template_string(
        _DASHBOARD,
        latest=_latest,
        log=list(_scan_log),
        now=datetime.now().strftime("%H:%M:%S"),
    )


@app.get("/latest")
def latest_json():
    """Return the latest scan result as JSON (useful for other IoT devices)."""
    if _latest is None:
        return jsonify({"status": "no scans yet"}), 200
    return jsonify(_latest), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
