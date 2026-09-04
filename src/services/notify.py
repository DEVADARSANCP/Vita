"""
Clinician notification — telling a human the case needs them.

Two rules carried over from ACPIA, for the same reasons and with more force
here, because this system's input comes from members of the public.

**The recipient is configuration, never an argument.** VITA can decide *that* a
case warrants notifying someone, and it can say why. It cannot decide *who*.
Addresses come from the hospital directory and the environment, and there is no
code path - none - that reads a recipient from model output or from anything a
patient typed. A system that ingests text from strangers and can also nominate
who receives outbound mail has a hole in it that no amount of prompt discipline
closes. Removing the capability is the only reliable fix.

**Dry run by default.** Nothing is sent unless `VITA_NOTIFY_ENABLED` is
explicitly true. In dry run the message is composed in full, recorded in the
outbox, and shown in the hospital dashboard - so the entire path can be
demonstrated and audited without anything crossing the network. Turning it on is
a deliberate act requiring SMTP configuration.

That default is also what makes the submission constraint hold: on a clean
machine with only a Gemini key, this module sends nothing and the feature still
demonstrates completely.

The message itself is assembled from the triage note. Nothing is generated for
it, so the clinician reads the same rule ids, the same unknowns and the same
escalation reasons that the dashboard shows.
"""

from __future__ import annotations

import logging
import os
import smtplib
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.message import EmailMessage
from typing import Any

logger = logging.getLogger(__name__)

#: Kept in memory and shown in the dashboard. Bounded, because an outbox that
#: grows without limit is a memory leak wearing a feature's clothes.
MAX_OUTBOX = 200


def _enabled() -> bool:
    return os.getenv("VITA_NOTIFY_ENABLED", "").strip().lower() in {"1", "true", "yes"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Notification:
    """One composed message, whether or not it was sent."""

    case_id: str
    channel: str
    recipient: str
    recipient_name: str
    subject: str
    body: str
    urgency: str
    at: str = field(default_factory=_now)
    dry_run: bool = True
    delivered: bool = False
    detail: str = ""

    def describe(self) -> str:
        if self.dry_run:
            return f"DRY RUN - would notify {self.recipient_name} <{self.recipient}>"
        if self.delivered:
            return f"delivered to {self.recipient_name} <{self.recipient}>"
        return f"FAILED - {self.detail}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "channel": self.channel,
            "recipient": self.recipient,
            "recipient_name": self.recipient_name,
            "subject": self.subject,
            "body": self.body,
            "urgency": self.urgency,
            "at": self.at,
            "dry_run": self.dry_run,
            "delivered": self.delivered,
            "detail": self.detail,
            "status": self.describe(),
        }


class Notifier:
    """Composes and records clinician notifications."""

    def __init__(self) -> None:
        self._outbox: list[Notification] = []
        self._lock = threading.Lock()

    # -- outbox ----------------------------------------------------------

    @property
    def outbox(self) -> list[Notification]:
        with self._lock:
            return list(reversed(self._outbox))

    def for_case(self, case_id: str) -> list[Notification]:
        return [n for n in self.outbox if n.case_id == case_id]

    def _record(self, notification: Notification) -> None:
        with self._lock:
            self._outbox.append(notification)
            if len(self._outbox) > MAX_OUTBOX:
                del self._outbox[: len(self._outbox) - MAX_OUTBOX]

    # -- sending ---------------------------------------------------------

    def notify_clinician(
        self,
        *,
        case_id: str,
        urgency: str,
        department: str,
        note_text: str,
        recipient: str,
        recipient_name: str,
        cited_rules: list[str],
        unknowns: list[str],
    ) -> Notification:
        """Compose the clinical escalation message, and send it only if enabled."""
        subject = f"VITA - {urgency} priority intake - {case_id}"

        header = [
            f"Case:       {case_id}",
            f"Urgency:    {urgency}",
            f"Department: {department}",
            f"Rules:      {', '.join(cited_rules) if cited_rules else 'none matched'}",
            f"Unknown:    {', '.join(unknowns) if unknowns else 'nothing outstanding'}",
            "",
            "This case was routed to you by an automated triage assistant. VITA does",
            "not diagnose. The full note, with the rule behind every line of it,",
            "follows.",
            "",
            "-" * 68,
            "",
        ]
        body = "\n".join(header) + note_text

        notification = Notification(
            case_id=case_id,
            channel="email",
            recipient=recipient,
            recipient_name=recipient_name,
            subject=subject,
            body=body,
            urgency=urgency,
            dry_run=not _enabled(),
        )

        if notification.dry_run:
            logger.info("notification for %s composed (dry run): %s", case_id, recipient)
            self._record(notification)
            return notification

        self._send(notification)
        self._record(notification)
        return notification

    def _send(self, notification: Notification) -> None:
        """Actually send. Only reachable when VITA_NOTIFY_ENABLED is true."""
        host = os.getenv("VITA_SMTP_HOST", "")
        port = int(os.getenv("VITA_SMTP_PORT", "587") or 587)
        user = os.getenv("VITA_SMTP_USER", "")
        password = os.getenv("VITA_SMTP_PASSWORD", "")
        sender = os.getenv("VITA_NOTIFY_FROM", user)

        if not host or not sender:
            notification.detail = "notification enabled but SMTP is not configured"
            logger.error(notification.detail)
            return

        message = EmailMessage()
        message["Subject"] = notification.subject
        message["From"] = sender
        message["To"] = notification.recipient
        message.set_content(notification.body)

        try:
            with smtplib.SMTP(host, port, timeout=15) as smtp:
                smtp.starttls()
                if user and password:
                    smtp.login(user, password)
                smtp.send_message(message)
            notification.delivered = True
            logger.info("notification for %s delivered to %s", notification.case_id, notification.recipient)
        except Exception as exc:  # noqa: BLE001 - delivery failure must not end the request
            notification.detail = f"{type(exc).__name__}: {exc}"
            logger.error("notification for %s failed: %s", notification.case_id, notification.detail)
