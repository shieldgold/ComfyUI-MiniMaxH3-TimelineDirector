"""Submit a ComfyUI API graph once and retain resumable execution evidence.

This uses only the standard library. It never cancels, reorders, frees models,
or restarts a service. An ambiguous HTTP outcome requires inspection rather
than an automatic retry that could duplicate a long GPU job.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone


def _now():
    return datetime.now(timezone.utc).isoformat()


def _write(path, value, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        if exclusive:
            os.link(temporary, path)
        else:
            os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _request(url, body=None):
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def _intent_from_bytes(endpoint, api_bytes):
    json.loads(api_bytes)
    return {"state": "intent", "utc": _now(), "endpoint": endpoint.rstrip("/"),
            "api_sha256": hashlib.sha256(api_bytes).hexdigest(),
            "client_id": str(uuid.uuid4())}


def make_intent(endpoint, api_path):
    return _intent_from_bytes(endpoint, Path(api_path).read_bytes())


def dispatch(endpoint, api_path, receipt_path):
    receipt_path = Path(receipt_path)
    api_bytes = Path(api_path).read_bytes()
    api = json.loads(api_bytes)
    intent = _intent_from_bytes(endpoint, api_bytes)
    try:
        _write(receipt_path, intent, exclusive=True)
    except FileExistsError:
        previous = json.loads(receipt_path.read_text())
        if previous.get("api_sha256") != intent["api_sha256"] or previous.get("endpoint") != intent["endpoint"]:
            raise RuntimeError("API or endpoint changed; use a new receipt")
        if previous.get("prompt_id"):
            return previous
        raise RuntimeError("Existing submission is uncertain or rejected; inspect the receipt and server before retrying")
    body = {"prompt": api, "client_id": intent["client_id"],
            "extra_data": {"receipt_client_id": intent["client_id"]}}
    try:
        response = _request(intent["endpoint"] + "/prompt", body)
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")
        try:
            detail = json.loads(detail)
        except json.JSONDecodeError:
            pass
        state = "uncertain" if error.code >= 500 else "rejected"
        _write(receipt_path, intent | {"state": state, "response": detail})
        raise RuntimeError(f"ComfyUI submission is {state}: {detail}") from error
    except Exception as error:
        _write(receipt_path, intent | {"state": "uncertain", "error": str(error)})
        raise RuntimeError("Submission outcome is uncertain; do not automatically resubmit") from error
    # ComfyUI can enqueue valid outputs while reporting errors on other outputs.
    # A returned prompt_id must always be retained and tracked as submitted.
    if not response.get("prompt_id"):
        _write(receipt_path, intent | {"state": "rejected", "response": response})
        raise RuntimeError(f"ComfyUI rejected the graph: {response}")
    result = intent | response | {"state": "submitted", "submitted_at": _now()}
    _write(receipt_path, result)
    return result


def _terminal(result):
    status = result.get("status", {})
    failures = [data for kind, data in status.get("messages", [])
                if kind in ("execution_error", "execution_interrupted")]
    return bool(status.get("completed") or failures or status.get("status_str") == "error")


def _finish(receipt_path, receipt, result):
    status = result.get("status", {})
    _write(receipt_path.with_suffix(".history.json"), {
        "prompt_id": receipt["prompt_id"], "api_sha256": receipt["api_sha256"],
        "endpoint": receipt["endpoint"], "result": result,
    })
    _write(receipt_path, receipt | {"state": status.get("status_str", "error"),
                                   "finished_at": receipt.get("finished_at", _now())})
    if status.get("status_str") != "success":
        raise RuntimeError(f"GPU job failed: {status}")
    return result


def wait_for_job(endpoint, receipt_path, poll_seconds=20):
    receipt_path = Path(receipt_path)
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("endpoint") != endpoint.rstrip("/"):
        raise RuntimeError("Endpoint does not match the receipt")
    prompt_id = receipt["prompt_id"]
    local_history = receipt_path.with_suffix(".history.json")
    if local_history.exists():
        cached = json.loads(local_history.read_text())
        identity_matches = all(cached.get(key) == receipt[key]
                               for key in ("prompt_id", "api_sha256", "endpoint"))
        result = cached.get("result", {})
        if identity_matches and _terminal(result):
            return _finish(receipt_path, receipt, result)
    while True:
        history = _request(endpoint.rstrip("/") + "/history/" + prompt_id)
        result = history.get(prompt_id)
        if result is not None:
            if _terminal(result):
                return _finish(receipt_path, receipt, result)
        print(json.dumps({"utc": _now(), "prompt_id": prompt_id, "state": "queued_or_running"}), flush=True)
        time.sleep(poll_seconds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--api", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--poll-seconds", default=20, type=float)
    args = parser.parse_args()
    if args.poll_seconds <= 0:
        parser.error("--poll-seconds must be positive")
    print(json.dumps(dispatch(args.endpoint, args.api, args.receipt), ensure_ascii=False), flush=True)
    if args.wait:
        result = wait_for_job(args.endpoint, args.receipt, args.poll_seconds)
        print(json.dumps({"status": result["status"], "outputs": result.get("outputs", {})}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
