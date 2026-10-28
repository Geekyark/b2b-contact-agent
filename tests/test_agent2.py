"""Agent 2 - B2B Decision-Maker Contact Finder: tests.

Run with:  python -m pytest tests/ -q
All tests run offline (no API calls). Discovery tests use the ``mock``
search provider and stub out Claude with ``FakeAnthropic``, which returns
fixed structured extractions snippets.
"""

from __future__ import annotations

import io
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pytest  # noqa: E402
from pydantic import ValidationError  # noqa: E402

from agent2.config import Settings  # noqa: E402
from agent2.discovery import run_discovery  # noqa: E402
from agent2.exporters import (  # noqa: E402
    EXPORT_COLUMNS,
    export_all,
    export_csv,
    export_excel,
    export_json,
    rows_from_results,
)
from agent2.inputs import (  # noqa: E402
    FIELD_ALIASES,
    LoadResult,
    load_companies,
    load_companies_from_csv_file,
    load_companies_from_dataframe,
    load_companies_from_json_file,
    load_companies_from_json_text,
    make_company_id,
    parse_company_dict,
    serialize_companies,
)
from agent2.models import (  # noqa: E402
    Agent1Company,
    CompanyContacts,
    Contact,
    ContactReviewState,
    EmailStatus,
    Evidence,
    PhoneStatus,
    looks_like_linkedin,
    normalize_linkedin,
    valid_email_shape,
    valid_phone_shape,
)
from agent2.search import (  # noqa: E402
    MockSearchProvider,
    SearchResult,
    SearchResponse,
    TavilySearchProvider,
    build_search_provider,
    query_variants,
)
from agent2.storage import Database, contact_key  # noqa: E402
from agent2.verification import (  # noqa: E402
    AbstractPhoneVerifier,
    NeverBounceVerifier,
    ZeroBounceVerifier,
    build_email_verifier,
    build_phone_verifier,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture()
def settings(tmp_path):
    s = Settings.from_env()
    s.search_provider = "mock"
    s.tavily_api_key = ""
    s.anthropic_api_key = ""
    s.database_path = str(tmp_path / "test.db")
    s.output_dir = str(tmp_path / "out")
    s.min_contacts_per_company = 1
    s.max_contacts_per_company = 5
    s.email_verification_provider = "none"
    s.phone_verification_provider = "none"
    return s


def make_company(**overrides) -> Agent1Company:
    base = dict(
        company_id="company-001",
        company_name="ExampleCorp",
        website="https://www.examplecorp.com",
        country="United Kingdom",
        city="London",
        industry="Software",
        company_size="51-200",
        products_services="Custom software development",
    )
    base.update(overrides)
    return Agent1Company(**base)


class FakeAnthropic:
    """Stub for the Anthropic client: returns fixed structured extractions.

    The default fixture cites the MockSearchProvider's real pages so the
    grounding path is exercised end-to-end offline.
    """

    def __init__(self, contacts: list[dict] | None = None):
        self.calls = []
        self.beta = self
        self.messages = self  # supports client.beta.messages.stream(...) style
        self.contacts = contacts or [
            {
                "name": "Ada Lovelace",
                "title": "Founder & CEO",
                "department": "Executive",
                "country": "United Kingdom",
                "linkedin_url": "https://www.linkedin.com/in/ada-lovelace",
                "work_email": "",
                "email_status": "not_found",
                "business_phone": "",
                "phone_status": "not_found",
                "role_relevance": "Founder and CEO with budget authority for IT services engagements.",
                "identity_source_url": "https://examplecorp.dev/about",
                "contact_source_url": "https://examplecorp.dev/team",
                "confidence": 0.85,
                "verification_notes": "Profile published on official company site; email not published.",
            },
            {
                "name": "Grace Hopper",
                "title": "CTO",
                "department": "Technology",
                "country": "United Kingdom",
                "linkedin_url": "https://www.linkedin.com/in/grace-hopper",
                "work_email": "grace@examplecorp.dev",
                "email_status": "published_in_source_not_independently_verified",
                "business_phone": "",
                "phone_status": "not_found",
                "role_relevance": "CTO owns engineering and vendor selection for technology services.",
                "identity_source_url": "https://examplecorp.dev/team",
                "contact_source_url": "https://examplecorp.dev/team",
                "confidence": 0.8,
                "verification_notes": "Email published on official team page; not independently verified.",
            },
        ]

    def messages(self):
        return self

    def stream(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeStream(self._build_message())

    def _build_message(self):
        from agent2.extraction import ExtractionResult, ExtractedContact
        parsed = ExtractionResult(
            contacts=[ExtractedContact(**c) for c in self.contacts],
            notes="fixed fixture",
        )
        klass = type("_Msg", (), {"stop_reason": "end_turn", "parsed_output": parsed})
        return klass()


class _FakeStream:
    """Mirrors MessageStreamManager: usable as a context manager that yields
    the final ParsedMessage."""

    def __init__(self, message):
        self._message = message

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return self._message


# ---------------------------------------------------------------------------
# Input adapters
# ---------------------------------------------------------------------------
class TestInputs:
    def test_parse_aliases(self):
        raw = {
            "companyName": "Acme",
            "id": "x-1",
            "domain": "acme.com",
            "country_name": "USA",
        }
        comp = parse_company_dict(raw)
        assert comp.company_name == "Acme"
        assert comp.company_id == "x-1"
        assert comp.website == "acme.com"
        assert comp.country == "USA"

    def test_missing_id_generated(self):
        comp = parse_company_dict({"company_name": "Acme Corp"})
        assert comp.company_id.startswith("acme-corp-")

    def test_missing_name(self):
        from agent2.inputs import parse_company_dict
        comp = parse_company_dict({"website": "x.com"})
        assert comp.company_name == ""
        assert comp.company_id  # generated

    def test_json_single_record_shape(self):
        text = json.dumps({"company": {"company_name": "One", "id": "1"}})
        result = load_companies_from_json_text(text)
        assert len(result.companies) == 1
        assert result.companies[0].company_name == "One"

    def test_json_raw_list_shape(self):
        text = json.dumps([{"company_name": "A", "id": "a"}, {"company_name": "B", "id": "b"}])
        result = load_companies_from_json_text(text)
        assert len(result.companies) == 2

    def test_json_invalid_reported(self):
        result = load_companies_from_json_text("{not json")
        assert result.errors
        assert not result.companies

    def test_csv_roundtrip(self, tmp_path):
        csv_text = (
            "company_id,company_name,website,country\n"
            "c1,Acme,acme.com,UK\n"
            "c2,Globex,globex.com,USA\n"
        )
        p = tmp_path / "companies.csv"
        p.write_text(csv_text, encoding="utf-8")
        result = load_companies_from_csv_file(p)
        assert len(result.companies) == 2
        assert result.companies[0].company_name == "Acme"

    def test_dataframe_roundtrip(self):
        import pandas as pd
        df = pd.DataFrame([{"company_name": "A", "id": "a"}, {"company_name": "B", "id": "b"}])
        result = load_companies_from_dataframe(df)
        assert len(result.companies) == 2

    def test_load_detects_csv_vs_json(self, tmp_path):
        p = tmp_path / "c.csv"
        p.write_text("company_name,company_id\nX,1\n", encoding="utf-8")
        result = load_companies(p)
        assert result.companies[0].company_name == "X"

    def test_company_id_generator(self):
        assert make_company_id("Acme Corp") == "acme-corp-" + make_company_id("Acme Corp").split("-")[-1]


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class TestModels:
    def test_contact_enum_defaults(self):
        c = Contact()
        assert c.email_status == EmailStatus.NOT_FOUND
        assert c.phone_status == PhoneStatus.NOT_FOUND
        assert c.review_state == ContactReviewState.PENDING

    def test_contact_set_enum(self):
        c = Contact(work_email="a@b.com", email_status="published_in_source_not_independently_verified")
        assert c.email_status == EmailStatus.PUBLISHED

    def test_invalid_email_status_rejected(self):
        with pytest.raises(ValidationError):
            Contact(email_status="bogus_status")

    def test_review_state_rejected_invalid(self):
        with pytest.raises(ValidationError):
            Contact(review_state="nope")

    def test_email_shape_validator(self):
        assert valid_email_shape("jane@acme.com")
        assert not valid_email_shape("not-an-email")
        assert not valid_email_shape("")

    def test_linkedin_normalizer(self):
        assert normalize_linkedin("linkedin.com/in/jane") == "https://linkedin.com/in/jane"
        assert normalize_linkedin("www.linkedin.com/in/jane") == "https://www.linkedin.com/in/jane"
        assert normalize_linkedin("https://www.linkedin.com/in/jane?ref=x") == "https://www.linkedin.com/in/jane"
        assert looks_like_linkedin("https://www.linkedin.com/in/jane")

    def test_phone_shape(self):
        assert valid_phone_shape("+44 20 7946 0958")
        assert not valid_phone_shape("n/a")


# ---------------------------------------------------------------------------
# Search providers
# ---------------------------------------------------------------------------
class TestSearch:
    def test_mock_provider_returns_results(self):
        prov = MockSearchProvider()
        resp = prov.search("ExampleCorp CEO")
        assert resp.results
        assert all(r.url for r in resp.results)

    def test_mock_matches_keywords(self):
        prov = MockSearchProvider()
        resp = prov.search("ExampleCorp CTO")
        assert any("linkedin.com" in r.url for r in resp.results)

    def test_query_variants_structure(self):
        variants = query_variants("Acme", "https://acme.com", ["CTO", "CEO"], 6)
        assert variants
        assert "Acme" in variants[0]
        assert "acme.com" in " ".join(variants)

    def test_tavily_provider_network_error_backs_off(self, monkeypatch):
        """A failing provider must return an error response, not raise."""
        import httpx

        class FailingTransport(httpx.BaseTransport):
            def handle_request(self, request):
                raise httpx.ConnectError("no network", request=request)

        settings = Settings.from_env()
        settings.tavily_api_key = "test-key"
        settings.tavily_max_retries = 1
        client = httpx.Client(transport=FailingTransport())
        provider = TavilySearchProvider(settings, client)
        resp = provider.search("anything")
        assert resp.error
        assert not resp.results


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
class TestVerificationMapping:
    def test_neverbounce_valid(self):
        class FakeResp:
            def raise_for_status(self):
                pass

            def json(self):
                return {"result": "valid"}

        v = NeverBounceVerifier("k")
        assert v.verify("a@b.com", _FakeClient(FakeResp())) == EmailStatus.VERIFIED

    def test_zerobounce_catchall(self):
        class FakeResp:
            def raise_for_status(self):
                pass

            def json(self):
                return {"status": "catch_all"}

        v = ZeroBounceVerifier("k")
        assert v.verify("a@b.com", _FakeClient(FakeResp())) == EmailStatus.CATCH_ALL

    def test_abstract_valid(self):
        class FakeResp:
            def raise_for_status(self):
                pass

            def json(self):
                return {"valid": True, "country": "GB"}

        v = AbstractPhoneVerifier("k")
        assert v.verify("+442079460958", _FakeClient(FakeResp())) == PhoneStatus.VERIFIED

    def test_no_provider_configured(self, settings):
        assert build_email_verifier(settings) is None
        assert build_phone_verifier(settings) is None


class _FakeClient:
    def __init__(self, resp):
        self._resp = resp

    def post(self, *a, **k):
        return self._resp

    def get(self, *a, **k):
        return self._resp


# ---------------------------------------------------------------------------
# Extraction grounding
# ---------------------------------------------------------------------------
class TestExtractionGrounding:
    def test_system_prompt_contains_evergreen_rules(self):
        from agent2.extraction import EXTRACTION_SYSTEM
        assert "never guess" in EXTRACTION_SYSTEM.lower()
        assert "generic" in EXTRACTION_SYSTEM.lower()

    def test_evidence_builder_rejects_unknown_url(self):
        from agent2.extraction import EvidenceBuilder
        eb = EvidenceBuilder(available_urls={"https://www.examplecorp.com/about"})
        ev = eb.check("email", "x@y.com", "https://evil.com/fake")
        assert ev is None

    def test_build_evidence_grounds_values(self):
        from agent2.discovery import _build_evidence
        from agent2.extraction import ExtractedContact
        ec = ExtractedContact(
            name="Ada Lovelace",
            title="CEO",
            linkedin_url="https://www.linkedin.com/in/ada",
            work_email="ada@examplecorp.com",
            identity_source_url="https://www.examplecorp.com/about",
            contact_source_url="https://www.examplecorp.com/contact",
        )
        proofs = _build_evidence(ec, {"https://www.examplecorp.com/about", "https://www.examplecorp.com/contact"}, {})
        assert any(p.fact_type == "email" and p.value == "ada@examplecorp.com" for p in proofs)
        assert any(p.fact_type == "identity" and p.value == "Ada Lovelace" for p in proofs)


# ---------------------------------------------------------------------------
# Discovery (offline, FakeAnthropic + mock search)
# ---------------------------------------------------------------------------
class TestDiscovery:
    def test_run_discovery_returns_contacts(self, settings):
        fake = FakeAnthropic()
        results = run_discovery(
            [make_company()],
            settings,
            anthropic_client=fake,
            search_provider=MockSearchProvider(),
        )
        assert len(results) == 1
        r = results[0]
        assert r.company_id == "company-001"
        assert r.error == ""
        assert r.contacts
        contact = r.contacts[0]
        assert contact.name  # identity present
        assert contact.email_status in (EmailStatus.PUBLISHED, EmailStatus.VERIFIED, EmailStatus.NOT_FOUND)

    def test_no_evidence_yields_no_contacts(self, settings):
        class EmptyProvider:
            name = "empty"

            def search(self, query):
                return SearchResponse(query=query, results=[])

            def close(self):
                pass

        fake = FakeAnthropic()
        results = run_discovery([make_company()], settings, anthropic_client=fake,
                                search_provider=EmptyProvider())
        assert results[0].contacts == []

    def test_hallucinated_contact_dropped(self, settings):
        """A contact whose sources are not in the search results must not survive."""
        fake = FakeAnthropic(contacts=[{
            "name": "Invented Person",
            "title": "CTO",
            "work_email": "invented@examplecorp.com",
            "identity_source_url": "https://not-in-search.example/person",
            "contact_source_url": "https://not-in-search.example/contact",
            "confidence": 0.9,
        }])
        results = run_discovery([make_company()], settings, anthropic_client=fake,
                                search_provider=MockSearchProvider())
        # identity URL was not returned by the provider -> grounded evidence empty
        assert results[0].contacts == []

    def test_dedupe_drops_duplicates(self, settings):
        from agent2.discovery import _dedupe_contacts
        c1 = Contact(name="Ada Lovelace", work_email="ada@examplecorp.com", source_urls=["a.com"])
        c2 = Contact(name="Ada Lovelace", work_email="ada@examplecorp.com", source_urls=["a.com", "b.com"])
        deduped = _dedupe_contacts([c1, c2])
        assert len(deduped) == 1
        assert deduped[0].source_urls == ["a.com", "b.com"]

    def test_stop_on_error(self, settings):
        class BoomProvider:
            name = "boom"

            def search(self, query):
                raise RuntimeError("provider down")

            def close(self):
                pass

        try:
            run_discovery([make_company()], settings, anthropic_client=FakeAnthropic(),
                          search_provider=BoomProvider(), stop_on_error=True)
            pytest.fail("expected RuntimeError")
        except RuntimeError:
            pass


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------
class TestStorage:
    def test_run_lifecycle(self, tmp_path):
        db = Database(str(tmp_path / "t.db"))
        db.start_run("r1", 2)
        db.finish_run("r1", 3)
        run = db.get_run("r1")
        assert run["status"] == "completed"
        assert run["contact_count"] == 3
        db.close()

    def test_company_contact_roundtrip(self, tmp_path):
        db = Database(str(tmp_path / "t.db"))
        comp = make_company()
        contact = Contact(name="Ada", work_email="ada@examplecorp.com")
        db.upsert_company("r1", comp)
        db.upsert_contact("r1", comp.company_id, contact)
        rows = db.list_contacts("r1")
        assert len([r for r in rows if r["name"] == "Ada"]) == 1
        db.close()

    def test_review_state_update(self, tmp_path):
        db = Database(str(tmp_path / "t.db"))
        comp = make_company()
        contact = Contact(name="Ada")
        db.upsert_company("r1", comp)
        db.upsert_contact("r1", comp.company_id, contact)
        key = contact_key("Ada", "", "")
        db.set_review_state("r1", comp.company_id, key, "approved")
        rows = db.list_contacts("r1")
        assert rows[0]["review_state"] == "approved"
        db.close()

    def test_contact_key_stable(self):
        assert contact_key("Ada", "", "") == contact_key("ada", "", "")


# ---------------------------------------------------------------------------
# Exporters
# ---------------------------------------------------------------------------
class TestExports:
    def make_results(self):
        c1 = Contact(
            name="Ada Lovelace", title="CEO", work_email="ada@examplecorp.com",
            email_status=EmailStatus.PUBLISHED, review_state=ContactReviewState.APPROVED,
            source_urls=["https://www.examplecorp.com/about"],
        )
        c2 = Contact(
            name="Grace Hopper", title="CTO", work_email="",
            email_status=EmailStatus.NOT_FOUND, review_state=ContactReviewState.PENDING,
        )
        r = CompanyContacts(company_id="c1", company_name="ExampleCorp", contacts=[c1, c2])
        return [r]

    def test_rows_one_per_contact_keeps_company_id(self):
        rows = rows_from_results(self.make_results())
        assert len(rows) == 2
        assert all(r["company_id"] == "c1" for r in rows)
        assert all(r["customer_id"] == "c1" for r in rows)
        assert all(col in rows[0] for col in EXPORT_COLUMNS)

    def test_export_approved_drops_rejected(self):
        results = self.make_results()
        results[0].contacts[0].review_state = ContactReviewState.REJECTED
        rows = rows_from_results(results, include_rejected=False)
        assert len(rows) == 1

    def test_export_csv(self, tmp_path):
        out = export_csv(self.make_results(), tmp_path / "contacts.csv")
        text = out.read_text(encoding="utf-8-sig")
        assert "Ada Lovelace" in text
        assert "company_id" in text.splitlines()[0]

    def test_export_excel(self, tmp_path):
        import pandas as pd
        out = export_excel(self.make_results(), tmp_path / "contacts.xlsx")
        df = pd.read_excel(out)
        assert list(df.columns) == EXPORT_COLUMNS

    def test_export_json_contains_schema(self, tmp_path):
        out = export_json(self.make_results(), tmp_path / "contacts.json")
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["companies"][0]["company_id"] == "c1"

    def test_export_all_writes_three_files(self, tmp_path):
        paths = export_all(self.make_results(), tmp_path / "out")
        assert set(paths) == {"csv", "xlsx", "json"}
        assert all(p.exists() for p in paths.values())


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
class TestConfig:
    def test_defaults(self):
        s = Settings()
        assert s.search_provider == "tavily"
        assert s.email_verification_provider == "none"

    def test_from_env_flags(self, monkeypatch):
        monkeypatch.setenv("SEARCH_PROVIDER", "mock")
        monkeypatch.setenv("EMAIL_VERIFICATION_PROVIDER", "neverbounce")
        s = Settings.from_env()
        assert s.search_provider == "mock"
        assert s.email_verification_provider == "neverbounce"

    def test_problems_for_discovery(self):
        s = Settings()
        s.anthropic_api_key = ""
        s.search_provider = "tavily"
        s.tavily_api_key = ""
        problems = s.problems_for_discovery()
        assert any("ANTHROPIC_API_KEY" in p for p in problems)
        assert any("TAVILY_API_KEY" in p for p in problems)

    def test_mock_mode_has_no_problems(self):
        s = Settings()
        s.anthropic_api_key = "x"
        s.search_provider = "mock"
        s.tavily_api_key = ""
        assert s.problems_for_discovery() == []