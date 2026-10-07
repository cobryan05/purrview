"""AuthGate: the optional auth gate in front of the whole app (WP-3).

Settings → Security stores an `AuthConfig` blob; Manager caches it and AuthGate
checks every request against Manager.require_auth / check_token / check_session.
Tests set Manager's cached config directly rather than going through the settings
round-trip — the same pattern test_own_status.py uses for _inference_url_override —
except where the round-trip itself is what's under test.

Two credentials, deliberately asymmetric: the API token authenticates machines and
is stored verbatim; the login password authenticates a person and is only ever
stored as an scrypt digest.
"""

import json

import pytest

from apps.helpers.authSettings import AUTH_SETTING_KEY, AuthConfig, hash_password, verify_password
from apps.image_tagging_server.manager import LoginRateLimiter
from apps.image_tagging_server.web import AUTH_COOKIE_NAME
from scripts.reset_login_password import clear_login_password

TOKEN = "s3cr3t"


def _enable(manager, token=TOKEN, password=None):
    """Point Manager's cached config at a token and, optionally, a password."""
    manager._auth = AuthConfig(
        require_auth=True,
        token=token,
        password_enabled=password is not None,
        password_hash=hash_password(password) if password is not None else "",
    )


class TestAuthOff:
    """The default. Every acceptance bullet in WP-3 starts from "byte-identical to
    today" — these are the tests that would catch a regression there."""

    async def test_unauthenticated_request_succeeds_when_auth_is_off(self, http_client):
        resp = await http_client.get("/")
        assert resp.status_code == 200

    async def test_api_route_unaffected_when_auth_is_off(self, http_client):
        resp = await http_client.get("/api/labels/")
        assert resp.status_code == 200


class TestAuthOn:
    async def test_request_without_a_token_is_rejected(self, http_client, manager):
        _enable(manager)
        resp = await http_client.get("/api/labels/")
        assert resp.status_code == 401

    async def test_bearer_token_is_accepted(self, http_client, manager):
        _enable(manager)
        resp = await http_client.get("/api/labels/", headers={"Authorization": f"Bearer {TOKEN}"})
        assert resp.status_code == 200

    async def test_wrong_bearer_token_is_rejected(self, http_client, manager):
        _enable(manager)
        resp = await http_client.get("/api/labels/", headers={"Authorization": "Bearer nope"})
        assert resp.status_code == 401

    async def test_query_param_token_is_accepted(self, http_client, manager):
        """The path an <img>/<video> tag actually uses — it can't set headers."""
        _enable(manager)
        resp = await http_client.get(f"/api/labels/?token={TOKEN}")
        assert resp.status_code == 200

    async def test_session_cookie_is_accepted(self, http_client, manager):
        _enable(manager)
        cookie = manager.session_cookie_value()
        resp = await http_client.get("/api/labels/", headers={"Cookie": f"{AUTH_COOKIE_NAME}={cookie}"})
        assert resp.status_code == 200

    async def test_the_raw_token_is_not_a_valid_cookie(self, http_client, manager):
        """The cookie carries a derived session value, not the token. A cookie that is
        also a usable Bearer credential is exactly what the token/password split is
        meant to prevent."""
        _enable(manager, password="hunter2")
        resp = await http_client.get("/api/labels/", headers={"Cookie": f"{AUTH_COOKIE_NAME}={TOKEN}"})
        assert resp.status_code == 401

    async def test_rotating_the_token_invalidates_existing_sessions(self, http_client, manager):
        """No session store exists, so the cookie is derived from the secrets; changing
        either one must log every browser out rather than leaving a cookie minted under
        the old credentials working forever."""
        _enable(manager)
        cookie = manager.session_cookie_value()
        assert (
            await http_client.get("/api/labels/", headers={"Cookie": f"{AUTH_COOKIE_NAME}={cookie}"})
        ).status_code == 200

        _enable(manager, token="rotated")
        resp = await http_client.get("/api/labels/", headers={"Cookie": f"{AUTH_COOKIE_NAME}={cookie}"})
        assert resp.status_code == 401

    async def test_html_request_without_a_token_redirects_to_login_not_401(self, http_client, manager):
        _enable(manager)
        resp = await http_client.get("/", headers={"Accept": "text/html"}, follow_redirects=False)
        assert resp.status_code == 303
        assert resp.headers["location"].startswith("/login")

    async def test_json_request_without_a_token_gets_401_not_a_redirect(self, http_client, manager):
        """HA's proxy calls and the card's fetches ask for JSON/images, not HTML —
        redirecting those into a login page they cannot render would just look like
        a broken request instead of a clear auth failure."""
        _enable(manager)
        resp = await http_client.get("/api/labels/", headers={"Accept": "application/json"}, follow_redirects=False)
        assert resp.status_code == 401

    async def test_login_page_itself_stays_reachable(self, http_client, manager):
        """Otherwise a wrong token is a permanent lockout — no way back to the form."""
        _enable(manager)
        resp = await http_client.get("/login")
        assert resp.status_code == 200

    async def test_toggling_require_auth_off_reopens_the_app_without_a_token(self, http_client, manager):
        """require_auth is a Manager property, not a one-time snapshot — flipping it
        (e.g. after a Settings save) must take effect on the very next request, not
        just after a restart."""
        _enable(manager)
        assert (await http_client.get("/api/labels/")).status_code == 401
        manager._auth = AuthConfig()
        assert (await http_client.get("/api/labels/")).status_code == 200

    async def test_require_auth_with_no_token_saved_fails_open(self, manager):
        """A toggle flipped on with an empty token must not be able to lock the
        operator out with no recovery path."""
        manager._auth = AuthConfig(require_auth=True, token="")
        assert manager.require_auth is False

    async def test_password_required_with_no_hash_falls_back_to_the_token(self, manager):
        """Same fail-open reasoning one level down: a password toggle with nothing
        behind it would make the login form unanswerable."""
        manager._auth = AuthConfig(require_auth=True, token=TOKEN, password_enabled=True, password_hash="")
        assert manager.password_enabled is False
        assert await manager.check_login(TOKEN) is True


class TestPasswordHashing:
    def test_hash_is_salted_so_the_same_password_hashes_differently(self):
        assert hash_password("hunter2") != hash_password("hunter2")

    def test_verify_accepts_the_password_and_rejects_others(self):
        stored = hash_password("hunter2")
        assert verify_password("hunter2", stored) is True
        assert verify_password("hunter3", stored) is False

    def test_plaintext_does_not_appear_in_the_stored_value(self):
        """The whole point of the split: nothing recoverable is written to settings."""
        assert "hunter2" not in hash_password("hunter2")

    def test_a_corrupt_stored_hash_rejects_rather_than_raising(self):
        assert verify_password("hunter2", "not-a-hash") is False


class TestLoginFlow:
    async def test_correct_token_signs_in_when_no_password_is_configured(self, http_client, manager):
        _enable(manager)

        resp = await http_client.post("/login", data={"token": TOKEN, "next": "/some/page"}, follow_redirects=False)
        assert resp.status_code == 303
        assert resp.headers["location"] == "/some/page"
        assert f"{AUTH_COOKIE_NAME}={manager.session_cookie_value()}" in resp.headers.get("set-cookie", "")

    async def test_wrong_token_does_not_set_a_cookie(self, http_client, manager):
        _enable(manager)

        resp = await http_client.post("/login", data={"token": "wrong"}, follow_redirects=False)
        assert resp.status_code == 303
        assert "/login" in resp.headers["location"]
        assert "set-cookie" not in resp.headers

    async def test_the_password_signs_in_once_one_is_configured(self, http_client, manager):
        _enable(manager, password="hunter2")

        resp = await http_client.post("/login", data={"token": "hunter2"}, follow_redirects=False)
        assert resp.status_code == 303
        assert resp.headers["location"] == "/"

    async def test_the_api_token_stops_signing_in_once_a_password_is_configured(self, http_client, manager):
        """The separation the feature exists for: Home Assistant's copy of the token
        authenticates API calls but must not open a browser session."""
        _enable(manager, password="hunter2")

        resp = await http_client.post("/login", data={"token": TOKEN}, follow_redirects=False)
        assert resp.status_code == 303
        assert "/login" in resp.headers["location"]
        assert "set-cookie" not in resp.headers

    async def test_the_token_still_authenticates_the_api_with_a_password_set(self, http_client, manager):
        _enable(manager, password="hunter2")
        resp = await http_client.get("/api/labels/", headers={"Authorization": f"Bearer {TOKEN}"})
        assert resp.status_code == 200

    async def test_off_site_next_is_not_followed(self, http_client, manager):
        """`//evil.example.com/x` is not caught by a bare startswith("/") check —
        browsers treat a leading // as protocol-relative. This is the regression that
        matters; a same-site next value passing through is the uninteresting half."""
        _enable(manager)

        resp = await http_client.post(
            "/login", data={"token": TOKEN, "next": "//evil.example.com/x"}, follow_redirects=False
        )
        assert resp.headers["location"] == "/"


class TestLogout:
    async def test_logout_clears_the_cookie(self, http_client, manager):
        _enable(manager)

        resp = await http_client.get(
            "/logout",
            headers={"Cookie": f"{AUTH_COOKIE_NAME}={manager.session_cookie_value()}"},
            follow_redirects=False,
        )
        assert resp.status_code == 303
        assert "Max-Age=0" in resp.headers.get("set-cookie", "")


class TestAuthConfigEndpoint:
    async def test_get_reports_state_without_leaking_the_password_hash(self, http_client, manager):
        _enable(manager, password="hunter2")

        resp = await http_client.get("/api/auth-config", headers={"Authorization": f"Bearer {TOKEN}"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["requireAuth"] is True
        assert data["passwordEnabled"] is True
        assert data["passwordSet"] is True
        assert "passwordHash" not in data
        assert manager.auth_config.password_hash not in resp.text

    async def test_save_hashes_the_password_and_applies_it_immediately(self, http_client, manager, monkeypatch):
        written = {}

        async def fake_put(url, json):
            written["value"] = json["value"]

            class _Res:
                def raise_for_status(self):
                    pass

            return _Res()

        monkeypatch.setattr(manager._test_client, "put", fake_put)

        resp = await http_client.post(
            "/api/auth-config",
            json={"requireAuth": True, "token": TOKEN, "passwordEnabled": True, "password": "hunter2"},
        )
        assert resp.status_code == 200
        assert resp.json()["passwordSet"] is True
        assert "hunter2" not in written["value"]
        assert manager.password_enabled is True
        assert await manager.check_login("hunter2") is True

    async def test_saving_with_a_blank_password_keeps_the_existing_one(self, http_client, manager, monkeypatch):
        """The panel never prefills the field, so a blank submission has to mean
        "leave it alone" — otherwise every unrelated save would wipe the password."""
        _enable(manager, password="hunter2")
        original = manager.auth_config.password_hash

        async def fake_put(url, json):
            class _Res:
                def raise_for_status(self):
                    pass

            return _Res()

        monkeypatch.setattr(manager._test_client, "put", fake_put)

        resp = await http_client.post(
            "/api/auth-config",
            json={"requireAuth": True, "token": TOKEN, "passwordEnabled": True, "password": ""},
            headers={"Authorization": f"Bearer {TOKEN}"},
        )
        assert resp.status_code == 200
        assert manager.auth_config.password_hash == original

    async def test_requiring_auth_without_a_token_is_refused(self, http_client, manager):
        resp = await http_client.post(
            "/api/auth-config",
            json={"requireAuth": True, "token": "", "passwordEnabled": False, "password": ""},
        )
        assert resp.status_code == 400
        assert resp.json()["status"] == "failure"

    async def test_requiring_a_password_without_one_set_is_refused(self, http_client, manager):
        resp = await http_client.post(
            "/api/auth-config",
            json={"requireAuth": True, "token": TOKEN, "passwordEnabled": True, "password": ""},
        )
        assert resp.status_code == 400

    async def test_a_failed_write_does_not_change_the_live_gate(self, manager, monkeypatch):
        """Manager caches the config, so applying it before the write lands would leave
        the running gate enforcing something that was never persisted."""

        async def failing_put(url, json):
            raise RuntimeError("db_server down")

        monkeypatch.setattr(manager._test_client, "put", failing_put)

        with pytest.raises(RuntimeError):
            await manager.save_auth_config(require_auth=True, token=TOKEN, password_enabled=False, password=None)
        assert manager.require_auth is False


class TestLoginRateLimiter:
    """The failed-login counter behind POST /login.

    Driven by an injected clock rather than by sleeping: the window is 60s, and a test
    that waited it out would be a minute of nothing.
    """

    def _limiter(self, **kwargs):
        self.now = 1000.0
        return LoginRateLimiter(clock=lambda: self.now, **kwargs)

    def test_failures_up_to_the_limit_are_allowed(self):
        limiter = self._limiter(max_failures=3)

        for _ in range(2):
            assert limiter.retry_after("1.2.3.4") == 0.0
            limiter.note_failure("1.2.3.4")

        assert limiter.retry_after("1.2.3.4") == 0.0, "the third attempt is still within the limit"

    def test_the_client_is_turned_away_once_the_limit_is_reached(self):
        limiter = self._limiter(max_failures=3, window_s=60.0)

        for _ in range(3):
            limiter.note_failure("1.2.3.4")

        assert limiter.retry_after("1.2.3.4") > 0

    def test_the_window_reopens(self):
        """The failure that would matter most: a window that never expires is a
        permanent lockout, and the only way back would be restarting the service."""
        limiter = self._limiter(max_failures=2, window_s=60.0)
        limiter.note_failure("1.2.3.4")
        limiter.note_failure("1.2.3.4")
        assert limiter.retry_after("1.2.3.4") > 0

        self.now += 60.1

        assert limiter.retry_after("1.2.3.4") == 0.0

    def test_the_wait_shrinks_as_the_oldest_failure_ages(self):
        limiter = self._limiter(max_failures=2, window_s=60.0)
        limiter.note_failure("1.2.3.4")
        limiter.note_failure("1.2.3.4")
        first = limiter.retry_after("1.2.3.4")

        self.now += 30.0

        assert limiter.retry_after("1.2.3.4") < first

    def test_a_success_forgets_the_failures(self):
        """The anti-lockout property. The key can only be the connecting address, so
        behind a proxy every browser shares one — whoever knows the password has to get
        in on the attempt they make."""
        limiter = self._limiter(max_failures=2)
        limiter.note_failure("1.2.3.4")
        limiter.note_failure("1.2.3.4")
        assert limiter.retry_after("1.2.3.4") > 0

        limiter.note_success("1.2.3.4")

        assert limiter.retry_after("1.2.3.4") == 0.0

    def test_clients_are_counted_separately(self):
        limiter = self._limiter(max_failures=2)
        limiter.note_failure("1.2.3.4")
        limiter.note_failure("1.2.3.4")

        assert limiter.retry_after("1.2.3.4") > 0
        assert limiter.retry_after("5.6.7.8") == 0.0, "one client's guesses must not block another"

    def test_the_table_does_not_grow_without_bound(self):
        """The key is whatever address connected, so the table is caller-driven."""
        limiter = self._limiter(max_failures=2, client_limit=10)

        for i in range(200):
            limiter.note_failure(f"10.0.0.{i}")

        assert len(limiter._failures) <= 10

    def test_expired_clients_are_dropped_entirely(self):
        limiter = self._limiter(max_failures=5, window_s=60.0)
        limiter.note_failure("1.2.3.4")

        self.now += 61.0
        limiter.note_failure("5.6.7.8")

        assert "1.2.3.4" not in limiter._failures


class TestLoginIsRateLimited:
    """POST /login, the one path AuthGate exempts and so the only one an
    unauthenticated caller can spend scrypt time on."""

    async def test_repeated_wrong_guesses_are_eventually_refused(self, http_client, manager):
        _enable(manager)

        last = None
        for _ in range(11):
            last = await http_client.post("/login", data={"token": "wrong"}, follow_redirects=False)

        assert last.status_code == 429
        assert int(last.headers["Retry-After"]) > 0

    async def test_the_credential_is_not_even_checked_once_refused(self, http_client, manager, monkeypatch):
        """The whole point: a refused caller must not reach the hash."""
        _enable(manager)
        for _ in range(11):
            await http_client.post("/login", data={"token": "wrong"}, follow_redirects=False)

        checked = []
        original = manager.check_login

        async def _spy(candidate):
            checked.append(candidate)
            return await original(candidate)

        monkeypatch.setattr(manager, "check_login", _spy)
        resp = await http_client.post("/login", data={"token": "wrong"}, follow_redirects=False)

        assert resp.status_code == 429
        assert checked == [], "a rate-limited attempt must not cost a credential check"

    async def test_the_right_credential_still_signs_in_after_some_failures(self, http_client, manager):
        """A limit that locked out the person who knows the password would be worse
        than no limit — behind a proxy every browser shares one key."""
        _enable(manager)
        for _ in range(9):
            await http_client.post("/login", data={"token": "wrong"}, follow_redirects=False)

        resp = await http_client.post("/login", data={"token": TOKEN}, follow_redirects=False)

        assert resp.status_code == 303
        assert resp.headers["location"] == "/"

    async def test_signing_in_clears_the_count_for_the_next_attempt(self, http_client, manager):
        """Nine typos, a sign-in, then nine more typos must still be nine — not
        eighteen. Without the reset the tenth failure overall is refused, and a long
        session of occasional fumbles would eventually block itself."""
        _enable(manager)
        for _ in range(9):
            await http_client.post("/login", data={"token": "wrong"}, follow_redirects=False)
        await http_client.post("/login", data={"token": TOKEN}, follow_redirects=False)

        last = None
        for _ in range(9):
            last = await http_client.post("/login", data={"token": "wrong"}, follow_redirects=False)

        assert last.status_code == 303, "a successful sign-in must forget the earlier failures"

    async def test_the_form_is_still_served_while_refused(self, http_client, manager):
        """GET /login costs nothing and must stay reachable, or a rate-limited client
        cannot even see why it was turned away."""
        _enable(manager)
        for _ in range(11):
            await http_client.post("/login", data={"token": "wrong"}, follow_redirects=False)

        resp = await http_client.get("/login")

        assert resp.status_code == 200


class TestPasswordRecovery:
    """`scripts/reset_login_password.py` is the only way back in once a password is
    lost — there is no email and deliberately no web route, since a web route would
    have to be reachable while logged out."""

    @staticmethod
    def _db(tmp_path, config):
        import sqlite3

        tmp_path.mkdir(parents=True, exist_ok=True)
        path = tmp_path / "db.sqlite"
        conn = sqlite3.connect(path)
        conn.execute("CREATE TABLE user_settings (key TEXT PRIMARY KEY, value TEXT)")
        if config is not None:
            conn.execute("INSERT INTO user_settings VALUES (?, ?)", (AUTH_SETTING_KEY, json.dumps(config)))
        conn.commit()
        conn.close()
        return path

    @staticmethod
    def _stored(path):
        import sqlite3

        row = (
            sqlite3.connect(path)
            .execute("SELECT value FROM user_settings WHERE key = ?", (AUTH_SETTING_KEY,))
            .fetchone()
        )
        return json.loads(row[0])

    def test_clearing_the_password_leaves_the_gate_and_token_alone(self, tmp_path):
        """A reset that also dropped require_auth would silently open the whole app to
        the network — the opposite of what the operator asked for."""
        path = self._db(
            tmp_path,
            {"require_auth": True, "token": "tok", "password_enabled": True, "password_hash": hash_password("x")},
        )

        clear_login_password(path)

        stored = self._stored(path)
        assert stored["password_enabled"] is False
        assert stored["password_hash"] == ""
        assert stored["require_auth"] is True
        assert stored["token"] == "tok"

    async def test_a_cleared_password_lets_the_token_sign_in_again(self, manager, tmp_path):
        """The recovery actually recovers: the config the script leaves behind is one
        `check_login` accepts the API token against."""
        path = self._db(
            tmp_path,
            {"require_auth": True, "token": TOKEN, "password_enabled": True, "password_hash": hash_password("x")},
        )
        clear_login_password(path)

        manager._auth = AuthConfig(**self._stored(path))
        assert await manager.check_login(TOKEN) is True

    def test_the_operator_is_warned_only_when_the_reset_leaves_the_app_open(self, tmp_path):
        """Clearing the password on a token-less install removes the only credential.
        That is the right trade against a permanent lockout, but it must not be
        silent — and the warning must not cry wolf on an install that stays gated."""
        open_install = self._db(
            tmp_path / "open",
            {"require_auth": False, "token": "", "password_enabled": True, "password_hash": hash_password("x")},
        )
        gated_install = self._db(
            tmp_path / "gated",
            {"require_auth": True, "token": "tok", "password_enabled": True, "password_hash": hash_password("x")},
        )

        assert "open to anyone" in clear_login_password(open_install)
        assert "open to anyone" not in clear_login_password(gated_install)

    def test_no_stored_settings_is_not_an_error(self, tmp_path):
        assert "no password" in clear_login_password(self._db(tmp_path, None)).lower()


class TestPasswordOnlyGatesOnItsOwn:
    """A password with no API token has to enforce by itself.

    The two switches are independent credentials, not a master and a sub-option:
    someone who wants a login password and has no use for the API should never have
    to generate a token to make the password take effect, and a stored
    "password required, gate off" is a password nobody is ever asked for.
    """

    def _password_only(self, manager, password="hunter2"):
        manager._auth = AuthConfig(
            require_auth=False,
            token="",
            password_enabled=True,
            password_hash=hash_password(password),
        )

    async def test_the_gate_enforces_with_a_password_and_no_token(self, manager):
        self._password_only(manager)
        assert manager.require_auth is True

    async def test_a_page_request_is_sent_to_the_login_form(self, http_client, manager):
        self._password_only(manager)
        resp = await http_client.get("/", headers={"Accept": "text/html"}, follow_redirects=False)
        assert resp.status_code == 303
        assert resp.headers["location"].startswith("/login")

    async def test_the_password_signs_in_and_the_session_works(self, http_client, manager):
        self._password_only(manager)

        resp = await http_client.post("/login", data={"token": "hunter2"}, follow_redirects=False)
        assert resp.status_code == 303
        assert resp.headers["location"] == "/"

        cookie = manager.session_cookie_value()
        assert (
            await http_client.get("/api/labels/", headers={"Cookie": f"{AUTH_COOKIE_NAME}={cookie}"})
        ).status_code == 200

    async def test_saving_a_password_without_a_token_is_allowed(self, manager, monkeypatch):
        """The panel must be able to persist this state — refusing it was the bug."""

        async def fake_put(url, json):
            class _Res:
                def raise_for_status(self):
                    pass

            return _Res()

        monkeypatch.setattr(manager._test_client, "put", fake_put)

        await manager.save_auth_config(require_auth=False, token="", password_enabled=True, password="hunter2")
        assert manager.require_auth is True
