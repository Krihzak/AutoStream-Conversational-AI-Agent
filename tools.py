"""Mock tools the agent can call. Kept in one module so the tool surface is
obvious and easy to extend."""

from __future__ import annotations

import re
from typing import Optional

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def mock_lead_capture(name: str, email: str, platform: str) -> str:
    """Simulates sending the lead to a CRM. Returns the confirmation string
    so the agent can surface it to the user."""
    message = f"Lead captured successfully: {name}, {email}, {platform}"
    print(message)
    return message


def validate_email(email: Optional[str]) -> bool:
    return bool(email and _EMAIL_RE.match(email.strip()))
