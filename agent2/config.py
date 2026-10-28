"""Application settings loaded from environment variables / a .env file.

Everything that is environment-dependent lives here so the rest of the
codebase is pure and testable. Copy `.env.example` to `.env` and fill in your
values.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv(override=True)

# The services offered by the business that Agent 2 prospects for. Used to
# explain *why* a contact is relevant, and to steer role targeting.
IT_SERVICES = [
    "Custom software development",
    "Web and mobile application development",
    "AI and automation",
    "Cloud computing and DevOps",
    "Cybersecurity",
    "QA and software testing",
    "IT outsourcing and dedicated developers",
    "API integrations and managed IT services",
]

# Target markets served by the business (used only for reporting/filtering).
TARGET_MARKETS = ["India", "USA", "United States", "UK", "United Kingdom",
                  "Canada", "Australia", "UAE", "United Arab Emirates"]

VALID_EMAIL_STATUSES = (
    "published_in_source_not_independently_verified",
    "verified_by_configured_provider",
    "invalid",
    "catch_all_or_uncertain",
    "not_found",
)

VALID_PHONE_STATUSES = (
    "published_in_source_not_independently_verified",
    "verified_by_configured_provider",
    "not_found",
)


def _get(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _get_bool(name: str, default: bool = False) -> bool:
    val = _get(name)
    if not val:
        return default
    return val.lower() in ("1", "true", "yes", "on")


@dataclass
class Settings:
    """Runtime configuration. Construct with ``Settings.from_env()``."""

    # Claude
    anthropic_api_key: str = ""
    claude_model: str = "claude-opus-5-5"
    claude_max_tokens: int = 16000

    # Search
    search_provider: str = "tavily"          # tavily | mock
    tavily_api_key: str = ""
    tavily_api_url: str = "https://api.tavily.com"
    tavily_max_results: int = 8
    search_timeout_seconds: float = 30.0
    search_max_retries: int = 3
    search_delay_seconds: float = 0.2        # courtesy delay between provider calls
    max_queries_per_company: int = 6

    # Discovery limits
    min_contacts_per_company: int = 1
    max_contacts_per_company: int = 5

    # Verification (optional; all default to disabled)
    email_verification_provider: str = "none"  # none | neverbounce | zerobounce
    neverbounce_api_key: str = ""
    zerobounce_api_key: str = ""
    phone_verification_provider: str = "none"  # none | abstract
    abstractapi_api_key: str = ""

    # Storage / output
    database_path: str = "data/agent2.db"
    output_dir: str = "data/output"

    # FastAPI server
    api_host: str = "127.0.0.1"
    api_port: int = 8123

    extra: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    @classmethod
    def from_env(cls) -> "Settings":
        s = cls()
        s.anthropic_api_key = _get("ANTHROPIC_API_KEY")
        s.claude_model = _get("CLAUDE_MODEL", s.claude_model)
        s.claude_max_tokens = int(_get("CLAUDE_MAX_TOKENS", str(s.claude_max_tokens)))

        s.search_provider = _get("SEARCH_PROVIDER", s.search_provider).lower()
        s.tavily_api_key = _get("TAVILY_API_KEY")
        s.tavily_api_url = _get("TAVILY_API_URL", s.tavily_api_url)
        s.tavily_max_results = int(_get("TAVILY_MAX_RESULTS", str(s.tavily_max_results)))
        s.search_timeout_seconds = float(_get("SEARCH_TIMEOUT_SECONDS", str(s.search_timeout_seconds)))
        s.search_max_retries = int(_get("SEARCH_MAX_RETRIES", str(s.search_max_retries)))
        s.search_delay_seconds = float(_get("SEARCH_DELAY_SECONDS", str(s.search_delay_seconds)))
        s.max_queries_per_company = int(_get("MAX_QUERIES_PER_COMPANY", str(s.max_queries_per_company)))

        s.min_contacts_per_company = int(_get("MIN_CONTACTS_PER_COMPANY", str(s.min_contacts_per_company)))
        s.max_contacts_per_company = int(_get("MAX_CONTACTS_PER_COMPANY", str(s.max_contacts_per_company)))

        s.email_verification_provider = _get("EMAIL_VERIFICATION_PROVIDER", s.email_verification_provider).lower()
        s.neverbounce_api_key = _get("NEVERBOUNCE_API_KEY")
        s.zerobounce_api_key = _get("ZEROBOUNCE_API_KEY")
        s.phone_verification_provider = _get("PHONE_VERIFICATION_PROVIDER", s.phone_verification_provider).lower()
        s.abstractapi_api_key = _get("ABSTRACTAPI_API_KEY")

        s.database_path = _get("DATABASE_PATH", s.database_path)
        s.output_dir = _get("OUTPUT_DIR", s.output_dir)
        s.api_host = _get("API_HOST", s.api_host)
        s.api_port = int(_get("API_PORT", str(s.api_port)))
        return s

    # ------------------------------------------------------------------
    def problems_for_discovery(self) -> list[str]:
        """Return a list of configuration problems that block live discovery."""
        problems: list[str] = []
        if not self.anthropic_api_key:
            problems.append(
                "ANTHROPIC_API_KEY is not set (or no `ant auth login` profile exists). "
                "Claude extraction is required to run discovery."
            )
        if self.search_provider == "tavily" and not self.tavily_api_key:
            problems.append(
                "TAVILY_API_KEY is not set. Searches will fail; set the key or use "
                "SEARCH_PROVIDER=mock for a deterministic sample run."
            )
        if self.search_provider not in ("tavily", "mock"):
            problems.append(
                f"SEARCH_PROVIDER={self.search_provider!r} is invalid; use 'tavily' or 'mock'."
            )
        if self.email_verification_provider not in ("none", "neverbounce", "zerobounce"):
            problems.append(
                f"EMAIL_VERIFICATION_PROVIDER={self.email_verification_provider!r} is invalid."
            )
        return problems