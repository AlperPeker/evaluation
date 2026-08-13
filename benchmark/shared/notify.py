import os
import urllib.request


def enabled() -> bool:
    return bool(os.environ.get("NTFY_TOPIC"))


def notify(message: str, title: str = "", tags: str = "", priority: str = "") -> None:
    """Send a push notification via ntfy. No-op unless NTFY_TOPIC is set.

    Reads NTFY_TOPIC (required) and NTFY_SERVER (default https://ntfy.sh).
    Never raises: a notification failure must not break the benchmark.
    """
    topic = os.environ.get("NTFY_TOPIC")
    if not topic:
        return

    server = os.environ.get("NTFY_SERVER", "https://ntfy.sh").rstrip("/")
    headers = {}
    if title:
        headers["Title"] = title
    if tags:
        headers["Tags"] = tags
    if priority:
        headers["Priority"] = priority

    req = urllib.request.Request(
        f"{server}/{topic}", data=message.encode("utf-8"), headers=headers, method="POST"
    )
    try:
        urllib.request.urlopen(req, timeout=10)
    except Exception as e:
        print(f"[notify] failed to send ntfy notification: {e}")
