"""Contact verification.

Discovery and verification are deliberately separate.

- Discovery marks every email/phone it finds as
  ``published_in_source_not_independently_verified``.
- Verification, when a provider is configured, upgrades that status to
  ``verified_by_configured_provider`` (or downgrades it to ``invalid`` /
  ``catch_all_or_uncertain`` / ``not_found``).

No provider is configured by default (```none```), so nothing here ever
touches the network unless the user opts in via the environment.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import httpx

from .config import Settings
from .models import EmailStatus, PhoneStatus

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Email verification
# ---------------------------------------------------------------------------
class NeverBounceVerifier:
    """NeverBounce single-email verification API.

    Request (POST https://api.neverbounce.com/v4/single/check):
        { "email": "person@company.com", "key": "<api_key>" }
    Response:
        200 {"result": "valid"|"invalid"|"catchall"|"unknown"|"failure", ...}
    """

    name = "neverbounce"

    def __init__(self, api_key: str, base_url: str = "https://api.neverbounce.com/v4"):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    def verify(self, email: str, client: Optional[httpx.Client] = None) -> EmailStatus:
        try:
            resp = (client or httpx.Client(timeout=20.0)).post(
                f"{self.base_url}/single/check",
                json={"email": email, "key": self.api_key},
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:  # network/provider errors: keep published status
            logger.warning("NeverBounce check failed for %s: %s", email, exc)
            return EmailStatus.PUBLISHED
        result = str(data.get("result", "")).lower()
        mapping = {
            "valid": EmailStatus.VERIFIED,
            "invalid": EmailStatus.INVALID,
            "catchall": EmailStatus.CATCH_ALL,
            "unknown": EmailStatus.PUBLISHED,
        }
        return mapping.get(result, EmailStatus.PUBLISHED)


class ZeroBounceVerifier:
    """ZeroBounce single-email verification API."""

    name = "zerobounce"

    def __init__(self, api_key: str, base_url: str = "https://api.zerobounce.net/v2"):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    def verify(self, email: str, client: Optional[httpx.Client] = None) -> EmailStatus:
        try:
            resp = (client or httpx.Client(timeout=20.0)).get(
                f"{self.base_url}/validate",
                params={"api_key": self.api_key, "email": email},
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:  # network/provider errors: keep published status
            logger.warning("ZeroBounce check failed for %s: %s", email, exc)
            return EmailStatus.PUBLISHED
        status = str(data.get("status", "")).lower()
        mapping = {
            "valid": EmailStatus.VERIFIED,
            "invalid": EmailStatus.INVALID,
            "catch_all": EmailStatus.CATCH_ALL,
            "do_not_mail": EmailStatus.INVALID,
            "spamtrap": EmailStatus.INVALID,
            "abuse": EmailStatus.INVALID,
        }
        return mapping.get(status, EmailStatus.PUBLISHED)


def build_email_verifier(settings: Settings):
    """Instantiate the configured email verifier, or None."""
    provider = settings.email_verification_provider
    if provider == "neverbounce":
        if not settings.neverbounce_api_key:
            logger.warning("EMAIL_VERIFICATION_PROVIDER=neverbounce but NEVERBOUNCE_API_KEY is not set; skipping verification.")
            return None
        return NeverBounceVerifier(settings.neverbounce_api_key)
    if provider == "zerobounce":
        if not settings.zerobounce_api_key:
            logger.warning("EMAIL_VERIFICATION_PROVIDER=zerobounce but ZEROBOUNCE_API_KEY is not set; skipping verification.")
            return None
        return ZeroBounceVerifier(settings.zerobounce_api_key)
    return None


# ---------------------------------------------------------------------------
# Phone verification
# ---------------------------------------------------------------------------
class AbstractPhoneVerifier:
    """Abstract API phone number validation."""

    name = "abstract"

    def __init__(self, api_key: str, base_url: str = "https://phonevalidation.abstractapi.com/v1"):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    def verify(self, phone: str, client: Optional[httpx.Client] = None) -> PhoneStatus:
        try:
            resp = (client or httpx.Client(timeout=20.0)).get(
                f"{self.base_url}/",
                params={"api_key": self.api_key, "phone": phone},
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:  # network/provider errors: keep published status
            logger.warning("Abstract phone check failed for %s: %s", phone, exc)
            return PhoneStatus.PUBLISHED
        if data.get("valid") is True:
            return PhoneStatus.VERIFIED
        return PhoneStatus.NOT_FOUND  # invalid / unknown number


def build_phone_verifier(settings: Settings):
    """Instantiate the configured phone verifier, or None."""
    provider = settings.phone_verification_provider
    if provider == "abstract":
        if not settings.abstractapi_api_key:
            logger.warning("PHONE_VERIFICATION_PROVIDER=abstract but ABSTRACTAPI_API_KEY is not set; skipping verification.")
            return None
        return AbstractPhoneVerifier(settings.abstractapi_api_key)
    return None


# ---------------------------------------------------------------------------
# Public orchestration
# ---------------------------------------------------------------------------
def verify_contact_details(run_id: str, company_id: str, contacts: list[Any],
                           settings: Settings,
                           http_client: Optional[httpx.Client] = None) -> None:
    """Run configured providers over the contacts of one batch.

    Mutates each contact's email/phone status in place (the caller persists).
    If no verifier is configured, statuses stay
    ``published_in_source_not_independently_verified``.
    """
    email_verifier = build_email_verifier(settings)
    phone_verifier = build_phone_verifier(settings)
    if email_verifier is None and phone_verifier is None:
        return

    for contact in contacts:
        if email_verifier is not None and contact.work_email and \
                contact.email_status == EmailStatus.PUBLISHED:
            new_status = email_verifier.verify(contact.work_email, http_client)
            contact.email_status = new_status
            contact.verification_notes = (
                contact.verification_notes + f" Email checked by {email_verifier.name}: {new_status.value}."
            ).strip()
        if phone_verifier is not None and contact.business_phone and \
                contact.phone_status == PhoneStatus.PUBLISHED:
            new_status = phone_verifier.verify(contact.business_phone, http_client)
            contact.phone_status = new_status
            contact.verification_notes = (
                contact.verification_notes + f" Phone checked by {phone_verifier.name}: {new_status.value}."
            ).strip()