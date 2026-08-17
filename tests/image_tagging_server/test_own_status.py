"""image_tagging_server answers /status about itself.

Every other service answers /status; this one 404'd. It hosts the entire
frontend and proxies every other service, so the process rendering the Service
Status panel was the one thing no health check could see — a container
healthcheck or uptime monitor pointed at the stack got a 404 and had to treat
"the frontend is up" as unknowable, and the panel could report "all services
ok" while saying nothing about itself.

The detail rows carry the *effective* upstream URLs, because those are
overridable at runtime: "inference is fine" and "inference is fine on the box
you are actually talking to" are different claims, and this panel is where you
go to tell them apart.
"""

import httpx


def _rows(status: dict) -> dict:
    """Flatten the status detail sections into {label: value}."""
    return {row["label"]: row["value"] for section in status["details"] for row in section["rows"]}


class TestOwnStatusShape:
    def test_reports_itself_as_ok(self, manager):
        status = manager.get_own_status()

        assert status["service"] == "image_tagging_server"
        assert status["status"] == "ok"

    def test_summary_says_what_this_service_is(self, manager):
        # The collapsed row is all most people read; "frontend" is the useful
        # word, not a count of nothing.
        assert "frontend" in manager.get_own_status()["summary"]

    def test_lists_every_upstream_it_proxies(self, manager):
        rows = _rows(manager.get_own_status())

        assert set(rows) == {
            "db_server",
            "tasks_server",
            "inference_server",
            "training_server",
            "relay_buffer_server",
        }

    def test_reports_the_configured_upstream_urls(self, manager):
        rows = _rows(manager.get_own_status())

        assert rows["db_server"] == "http://db:8003"
        assert rows["tasks_server"] == "http://tasks:8002"
        assert rows["training_server"] == "http://training:8005"

    def test_an_override_is_reflected_in_the_rows(self, manager):
        # The whole reason to show effective URLs: an override pointed at the
        # wrong host is otherwise invisible.
        manager._inference_url_override = "http://remote-gpu:8001"
        status = manager.get_own_status()

        assert _rows(status)["inference_server"] == "http://remote-gpu:8001"
        assert "override" in status["summary"]

    def test_no_override_is_not_announced(self, manager):
        assert "override" not in manager.get_own_status()["summary"]


class TestOwnStatusEndpoint:
    async def test_status_route_answers_200(self, http_client):
        resp = await http_client.get("/status")

        assert resp.status_code == 200
        body = resp.json()
        assert body["service"] == "image_tagging_server"
        assert body["status"] == "ok"


class TestIncludedInTheAggregate:
    async def test_it_appears_in_its_own_service_list(self, manager, monkeypatch):
        """The panel lists what the frontend proxies; it must also list the
        frontend, or 'all services ok' is a claim about everything except the
        thing making it."""

        async def _refused(*_a, **_kw):
            raise httpx.ConnectError("nothing is listening in a unit test")

        monkeypatch.setattr(manager._test_client, "get", _refused)

        services = await manager.get_all_service_status()
        names = [s["service"] for s in services]

        assert "image_tagging_server" in names
        # Itself first: it is the process rendering the panel, not one of the
        # things it proxies to.
        assert names[0] == "image_tagging_server"
        # And it does not displace the upstreams it was already reporting.
        assert {"db_server", "tasks_server", "inference_server"} <= set(names)

    async def test_its_own_row_is_ok_even_when_every_upstream_is_down(self, manager, monkeypatch):
        """Reporting itself must not depend on a probe: the case where every
        backend is unreachable is exactly when you need to know the frontend
        is the part still answering."""

        async def _refused(*_a, **_kw):
            raise httpx.ConnectError("nothing is listening in a unit test")

        monkeypatch.setattr(manager._test_client, "get", _refused)

        services = await manager.get_all_service_status()
        own = next(s for s in services if s["service"] == "image_tagging_server")

        assert own["status"] == "ok"
