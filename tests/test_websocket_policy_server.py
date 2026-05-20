from __future__ import annotations


def test_route_message_returns_traceback_for_policy_error():
    from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer

    class FailingPolicy:
        def predict_action(self, **_payload):
            raise ValueError("synthetic inference failure")

    server = WebsocketPolicyServer(FailingPolicy())
    response = server._route_message({"type": "infer", "payload": {"examples": []}})

    assert response["ok"] is False
    assert response["error"]["message"] == "synthetic inference failure"
    assert "ValueError: synthetic inference failure" in response["error"]["traceback"]
