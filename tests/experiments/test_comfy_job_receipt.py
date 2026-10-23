"""Exercise queue submission against a local HTTP server, without a GPU."""
import importlib.util
import hashlib
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


MODULE = Path(__file__).with_name("comfy_job_receipt.py")


class ReceiptTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location("job_receipt", MODULE)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.api = root / "api.json"
        self.api.write_text(json.dumps({"1": {"class_type": "SaveVideo", "inputs": {}}}))
        self.receipt = root / "dispatch.json"
        self.posts = []
        self.reject = False
        self.server_error = False
        self.partial_errors = False
        self.prompt_id = "our-prompt"
        parent = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def reply(self, value, status=200):
                data = json.dumps(value).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_POST(self):
                parent.posts.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                if parent.server_error:
                    self.reply({"error": "gateway_failure"}, 500)
                elif parent.reject:
                    self.reply({"error": "invalid_prompt", "node_errors": {"1": "missing input"}}, 400)
                else:
                    errors = {"unused-output": "missing input"} if parent.partial_errors else {}
                    self.reply({"prompt_id": parent.prompt_id, "number": 4, "node_errors": errors})

            def do_GET(self):
                self.reply({parent.prompt_id: {"status": {"completed": True, "status_str": "success"}, "outputs": {}}})

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        self.endpoint = f"http://127.0.0.1:{server.server_port}"

    def test_resume_does_not_submit_twice(self):
        first = self.module.dispatch(self.endpoint, self.api, self.receipt)
        second = self.module.dispatch(self.endpoint, self.api, self.receipt)
        self.assertEqual(first["prompt_id"], second["prompt_id"])
        self.assertEqual(len(self.posts), 1)
        result = self.module.wait_for_job(self.endpoint, self.receipt, poll_seconds=0)
        self.assertEqual(result["status"]["status_str"], "success")

    def test_changed_api_cannot_reuse_a_receipt(self):
        self.module.dispatch(self.endpoint, self.api, self.receipt)
        self.api.write_text("{}")
        with self.assertRaisesRegex(RuntimeError, "changed"):
            self.module.dispatch(self.endpoint, self.api, self.receipt)
        self.assertEqual(len(self.posts), 1)

    def test_uncertain_submission_is_never_retried_automatically(self):
        state = self.module.make_intent(self.endpoint, self.api)
        state["state"] = "uncertain"
        self.receipt.write_text(json.dumps(state))
        with self.assertRaisesRegex(RuntimeError, "uncertain"):
            self.module.dispatch(self.endpoint, self.api, self.receipt)
        self.assertEqual(self.posts, [])

    def test_http_rejection_is_recorded(self):
        self.reject = True
        with self.assertRaisesRegex(RuntimeError, "invalid_prompt"):
            self.module.dispatch(self.endpoint, self.api, self.receipt)
        data = json.loads(self.receipt.read_text())
        self.assertEqual(data["state"], "rejected")
        self.assertIn("node_errors", data["response"])

    def test_accepted_partial_graph_retains_prompt_id(self):
        self.partial_errors = True
        first = self.module.dispatch(self.endpoint, self.api, self.receipt)
        self.assertEqual(first["prompt_id"], "our-prompt")
        self.assertTrue(first["node_errors"])
        self.module.dispatch(self.endpoint, self.api, self.receipt)
        self.assertEqual(len(self.posts), 1)

    def test_server_failure_is_an_uncertain_submission(self):
        self.server_error = True
        with self.assertRaisesRegex(RuntimeError, "uncertain"):
            self.module.dispatch(self.endpoint, self.api, self.receipt)
        self.assertEqual(json.loads(self.receipt.read_text())["state"], "uncertain")

    def test_completed_local_result_survives_cleared_server_history(self):
        self.module.dispatch(self.endpoint, self.api, self.receipt)
        result = self.module.wait_for_job(self.endpoint, self.receipt, poll_seconds=0)
        with patch.object(self.module, "_request", side_effect=AssertionError("must reuse completed local evidence")):
            resumed = self.module.wait_for_job(self.endpoint, self.receipt, poll_seconds=0)
        self.assertEqual(resumed, result)

    def test_receipt_hash_describes_the_submitted_snapshot(self):
        original_bytes = self.api.read_bytes()
        original_write = self.module._write

        def change_file_after_claim(path, value, exclusive=False):
            original_write(path, value, exclusive)
            if exclusive:
                self.api.write_text(json.dumps({"new": {"class_type": "SaveImage", "inputs": {}}}))

        with patch.object(self.module, "_write", side_effect=change_file_after_claim):
            result = self.module.dispatch(self.endpoint, self.api, self.receipt)
        self.assertEqual(self.posts[0]["prompt"], json.loads(original_bytes))
        self.assertEqual(result["api_sha256"], hashlib.sha256(original_bytes).hexdigest())

    def test_stale_history_cannot_complete_a_new_submission(self):
        self.module.dispatch(self.endpoint, self.api, self.receipt)
        self.module.wait_for_job(self.endpoint, self.receipt, poll_seconds=0)
        self.receipt.unlink()  # A user removed the receipt, but left its cache.
        self.prompt_id = "new-prompt"
        self.module.dispatch(self.endpoint, self.api, self.receipt)
        new_result = {"status": {"completed": True, "status_str": "success"}, "outputs": {"new": True}}
        with patch.object(self.module, "_request", return_value={"new-prompt": new_result}) as request:
            result = self.module.wait_for_job(self.endpoint, self.receipt, poll_seconds=0)
        self.assertEqual(result, new_result)
        request.assert_called_once()


if __name__ == "__main__":
    unittest.main()
