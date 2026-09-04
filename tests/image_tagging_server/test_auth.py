"""AuthGate: the optional shared-token gate in front of the whole app (WP-3).

Settings → Security stores {"require_auth": bool, "token": str}; Manager caches it
and AuthGate checks every request against Manager.require_auth / Manager.check_token.
Tests set Manager's private state directly rather than going through the settings
round-trip — the same pattern test_own_status.py uses for _inference_url_override —
since the settings read/write path itself belongs to serverUrls-style tests, not here.
"""

from apps.image_tagging_server.web import AUTH_COOKIE_NAME


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
    def _enable(self, manager, token="s3cr3t"):
        manager._require_auth = True
        manager._auth_token = token

    async def test_request_without_a_token_is_rejected(self, http_client, manager):
        self._enable(manager)
        resp = await http_client.get("/api/labels/")
        assert resp.status_code == 401

    async def test_bearer_token_is_accepted(self, http_client, manager):
        self._enable(manager)
        resp = await http_client.get("/api/labels/", headers={"Authorization": "Bearer s3cr3t"})
        assert resp.status_code == 200

    async def test_wrong_bearer_token_is_rejected(self, http_client, manager):
        self._enable(manager)
        resp = await http_client.get("/api/labels/", headers={"Authorization": "Bearer nope"})
        assert resp.status_code == 401

    async def test_query_param_token_is_accepted(self, http_client, manager):
        """The path an <img>/<video> tag actually uses — it can't set headers."""
        self._enable(manager)
        resp = await http_client.get("/api/labels/?token=s3cr3t")
        assert resp.status_code == 200

    async def test_cookie_token_is_accepted(self, http_client, manager):
        self._enable(manager)
        resp = await http_client.get("/api/labels/", headers={"Cookie": f"{AUTH_COOKIE_NAME}=s3cr3t"})
        assert resp.status_code == 200

    async def test_html_request_without_a_token_redirects_to_login_not_401(self, http_client, manager):
        self._enable(manager)
        resp = await http_client.get("/", headers={"Accept": "text/html"}, follow_redirects=False)
        assert resp.status_code == 303
        assert resp.headers["location"].startswith("/login")

    async def test_json_request_without_a_token_gets_401_not_a_redirect(self, http_client, manager):
        """HA's proxy calls and the card's fetches ask for JSON/images, not HTML —
        redirecting those into a login page they cannot render would just look like
        a broken request instead of a clear auth failure."""
        self._enable(manager)
        resp = await http_client.get("/api/labels/", headers={"Accept": "application/json"}, follow_redirects=False)
        assert resp.status_code == 401

    async def test_login_page_itself_stays_reachable(self, http_client, manager):
        """Otherwise a wrong token is a permanent lockout — no way back to the form."""
        self._enable(manager)
        resp = await http_client.get("/login")
        assert resp.status_code == 200

    async def test_toggling_require_auth_off_reopens_the_app_without_a_token(self, http_client, manager):
        """require_auth is a Manager property, not a one-time snapshot — flipping it
        (e.g. after a Settings save calls reload_auth_config) must take effect on the
        very next request, not just after a restart."""
        self._enable(manager)
        assert (await http_client.get("/api/labels/")).status_code == 401
        manager._require_auth = False
        assert (await http_client.get("/api/labels/")).status_code == 200

    async def test_require_auth_with_no_token_saved_fails_open(self, manager):
        """Manager.require_auth, not the raw _require_auth flag — see its docstring.
        A toggle flipped on with an empty token must not be able to lock the operator
        out with no recovery path."""
        manager._require_auth = True
        manager._auth_token = ""
        assert manager.require_auth is False


class TestLoginFlow:
    async def test_correct_token_sets_a_cookie_and_redirects_to_next(self, http_client, manager):
        manager._require_auth = True
        manager._auth_token = "s3cr3t"

        resp = await http_client.post("/login", data={"token": "s3cr3t", "next": "/some/page"}, follow_redirects=False)
        assert resp.status_code == 303
        assert resp.headers["location"] == "/some/page"
        assert f"{AUTH_COOKIE_NAME}=s3cr3t" in resp.headers.get("set-cookie", "")

    async def test_wrong_token_does_not_set_a_cookie(self, http_client, manager):
        manager._require_auth = True
        manager._auth_token = "s3cr3t"

        resp = await http_client.post("/login", data={"token": "wrong"}, follow_redirects=False)
        assert resp.status_code == 303
        assert "/login" in resp.headers["location"]
        assert "set-cookie" not in resp.headers

    async def test_off_site_next_is_not_followed(self, http_client, manager):
        """`//evil.example.com/x` is not caught by a bare startswith("/") check —
        browsers treat a leading // as protocol-relative. This is the regression that
        matters; a same-site next value passing through is the uninteresting half."""
        manager._require_auth = True
        manager._auth_token = "s3cr3t"

        resp = await http_client.post(
            "/login", data={"token": "s3cr3t", "next": "//evil.example.com/x"}, follow_redirects=False
        )
        assert resp.headers["location"] == "/"


class TestLogout:
    async def test_logout_clears_the_cookie(self, http_client, manager):
        manager._require_auth = True
        manager._auth_token = "s3cr3t"

        resp = await http_client.get(
            "/logout", headers={"Cookie": f"{AUTH_COOKIE_NAME}=s3cr3t"}, follow_redirects=False
        )
        assert resp.status_code == 303
        assert "Max-Age=0" in resp.headers.get("set-cookie", "")


class TestReloadAuthConfigEndpoint:
    async def test_reload_endpoint_calls_manager_and_reports_new_state(self, http_client, manager, monkeypatch):
        async def fake_reload():
            manager._require_auth = True
            manager._auth_token = "s3cr3t"

        monkeypatch.setattr(manager, "reload_auth_config", fake_reload)

        resp = await http_client.post("/api/reload-auth-config")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert data["requireAuth"] is True
