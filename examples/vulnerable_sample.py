"""Deliberately vulnerable sample used to verify the security review detects real issues.

Do not import this. It exists only as the positive control for a scanner test.
"""

import os
import sqlite3
import subprocess

from flask import Flask, request

app = Flask(__name__)


@app.route("/user")
def get_user():
    # Unparameterised SQL built from a request parameter.
    user_id = request.args.get("id")
    conn = sqlite3.connect("app.db")
    return conn.execute(
        "SELECT name, email FROM users WHERE id = '" + user_id + "'"
    ).fetchall()


@app.route("/ping")
def ping():
    # Request data reaches a shell.
    host = request.args.get("host")
    return subprocess.check_output("ping -c 1 " + host, shell=True)


@app.route("/download")
def download():
    # No containment on a caller-supplied path.
    name = request.args.get("name")
    with open(os.path.join("/var/data", name)) as handle:
        return handle.read()
