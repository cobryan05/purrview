"""Clear the login password so a locked-out operator can get back in.

    docker exec -w /project purrview_mono_dev \
        python scripts/reset_login_password.py

There is no email recovery and deliberately no web route for this: once a login
password is set, `/login` stops accepting the API token, which is the whole point of
the split (Home Assistant's copy of the token must not open a browser session). That
leaves no in-band way back if the password is lost, so the way back is out of band —
shell access to the host, which whoever installed PurrView has by definition.

This only clears `password_enabled` and `password_hash`. `require_auth` and the API
token are left alone, so the gate stays on and the next sign-in falls back to the
token; set a new password from Settings → Security.

`image_tagging_server` caches the settings blob, so restart it afterwards (or restart
the container) for the change to take effect.
"""

import argparse
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from apps import DB_DIR  # noqa: E402
from apps.helpers.authSettings import AUTH_SETTING_KEY  # noqa: E402


def clear_login_password(db_path: Path) -> str:
    """Turn the login password off in the settings blob.

    `require_auth` and the API token are deliberately untouched: the gate stays on and
    the next sign-in falls back to the token, rather than this quietly opening the
    whole app to the network as a side effect of a password reset.

    Args:
        db_path (Path): Path to db.sqlite.

    Returns:
        str: A line describing what happened, for the operator.
    """
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute("SELECT value FROM user_settings WHERE key = ?", (AUTH_SETTING_KEY,)).fetchone()
        if not row or not row[0]:
            return "No security settings stored — no password to clear."

        config = json.loads(row[0])
        if not config.get("password_enabled") and not config.get("password_hash"):
            return "No login password is set."

        config["password_enabled"] = False
        config["password_hash"] = ""
        conn.execute("UPDATE user_settings SET value = ? WHERE key = ?", (json.dumps(config), AUTH_SETTING_KEY))
        conn.commit()
    finally:
        conn.close()
    return (
        "Login password cleared.\n"
        "Restart image_tagging_server (supervisord UI at :9001, or restart the container),\n"
        "then sign in with the API token and set a new password in Settings → Security."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", type=Path, default=DB_DIR / "db.sqlite", help="Path to db.sqlite.")
    args = parser.parse_args()

    if not args.db.exists():
        print(f"No database at {args.db}", file=sys.stderr)
        return 1

    print(clear_login_password(args.db))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
