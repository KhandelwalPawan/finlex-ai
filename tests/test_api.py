from __future__ import annotations

import json
import unittest
from unittest.mock import MagicMock, patch


class ApiHealthTest(unittest.TestCase):
    def _get_app(self):
        # Import inside test to avoid module-level side effects
        import api as api_module
        return api_module.app

    def test_health_ok_returns_200(self):
        from starlette.testclient import TestClient
        with patch("api.run_healthcheck", return_value={"status": "ok", "config": {}}):
            with patch("api.chain"):
                import api as api_module
                client = TestClient(api_module.app, raise_server_exceptions=True)
                response = client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_health_degraded_returns_503(self):
        from starlette.testclient import TestClient
        with patch("api.run_healthcheck", return_value={"status": "degraded", "config": {}}):
            import api as api_module
            client = TestClient(api_module.app)
            response = client.get("/health")
        self.assertEqual(response.status_code, 503)

    def test_chat_missing_question_returns_400(self):
        from starlette.testclient import TestClient
        with patch("api.run_healthcheck", return_value={"status": "ok", "config": {}}):
            import api as api_module
            client = TestClient(api_module.app)
            response = client.post("/chat", json={})
        self.assertEqual(response.status_code, 400)
        self.assertIn("question", response.json()["error"])

    def test_chat_invalid_json_returns_400(self):
        from starlette.testclient import TestClient
        import api as api_module
        client = TestClient(api_module.app)
        response = client.post(
            "/chat",
            content=b"not json",
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(response.status_code, 400)

    def test_chat_question_too_long_returns_422(self):
        from starlette.testclient import TestClient
        import api as api_module
        client = TestClient(api_module.app)
        long_q = "x" * 501
        response = client.post("/chat", json={"question": long_q})
        self.assertEqual(response.status_code, 422)

    def test_chat_success(self):
        from starlette.testclient import TestClient
        mock_result = {
            "answer": "Test answer",
            "citations": [],
            "sources": [],
            "confidence": "high",
        }
        import api as api_module
        api_module._chain = MagicMock()
        api_module._chain.invoke.return_value = mock_result
        client = TestClient(api_module.app)
        response = client.post("/chat", json={"question": "What does the document say?"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["answer"], "Test answer")
        # Reset
        api_module._chain = None

    def test_sources_endpoint_returns_list(self):
        from starlette.testclient import TestClient
        import api as api_module
        api_module._chain = MagicMock()
        api_module._chain.source_inventory.return_value = [
            {"source": "doc.pdf", "chunks": 10, "pages": 3}
        ]
        client = TestClient(api_module.app)
        response = client.get("/sources")
        self.assertEqual(response.status_code, 200)
        self.assertIn("sources", response.json())
        api_module._chain = None

    def test_chat_timeout_returns_504(self):
        from starlette.testclient import TestClient
        import api as api_module
        api_module._chain = MagicMock()
        api_module._chain.invoke.side_effect = TimeoutError("Model timed out")
        client = TestClient(api_module.app)
        response = client.post("/chat", json={"question": "What is this?"})
        self.assertEqual(response.status_code, 504)
        api_module._chain = None


if __name__ == "__main__":
    unittest.main()
