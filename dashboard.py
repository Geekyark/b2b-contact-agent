"""Agent 2 - Streamlit dashboard.

Workflow:
1. Connect to the local SQLite database (auto-created on first run).
2. Upload company data from Agent 1 (JSON or CSV).
3. Start contact discovery (requires the backend configuration from .env;
   without a Claude key or search key, use SEARCH_PROVIDER=mock to demo).
4. Browse contacts grouped by company; filter by country, title,
   verification status and review state; open source links.
5. Review / edit / approve / reject individual contacts.
6. Export approved contacts to CSV / Excel / JSON, or export everything.

Discovery never sends emails or LinkedIn messages - it only researches.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import sys
import uuid
from pathlib import Path

import pandas as pd
import streamlit as st

# Allow running from the repo root without installation.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent2.config import IT_SERVICES, Settings
from agent2.dashboard_utils import (
    load_database,
    start_discovery,
)
from agent2.exporters import EXPORT_COLUMNS
from agent2.inputs import load_companies_from_dataframe, load_companies_from_json_text
from agent2.models import ContactReviewState, EmailStatus, PhoneStatus
from agent2.storage import contact_key

st.set_page_config(
    page_title="Agent 2 - B2B Contact Finder",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_TAGLINE = "Decision-maker discovery for IT services - evidence-backed, never invented."


@st.cache_resource
def get_database() -> "Database":
    settings = Settings.from_env()
    return load_database(settings)


@st.cache_resource
def get_settings():
    return Settings.from_env()


@st.cache_data(ttl=60)
def contacts_dataframe(run_id, company_id, review_state, email_status, country, title_query) -> pd.DataFrame:
    db = get_database()
    rows = db.list_contacts(
        run_id=run_id or None,
        company_id=company_id or None,
        review_state=review_state or None,
        email_status=email_status or None,
        country=country or None,
        title_contains=title_query or None,
    )
    return pd.DataFrame(rows)


def fmt_date(value) -> str:
    if not value:
        return ""
    try:
        return pd.to_datetime(value).strftime("%Y-%m-%d")
    except Exception:
        return str(value)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🔎 Agent 2")
    st.caption(APP_TAGLINE)

    settings = get_settings()

    st.markdown("### Configuration")
    st.write(f"**Model:** `{settings.claude_model}`")
    st.write(f"**Search provider:** `{settings.search_provider}`")
    st.write(f"**Email verification:** `{settings.email_verification_provider}`")
    st.write(f"**Phone verification:** `{settings.phone_verification_provider}`")

    problems = settings.problems_for_discovery()
    offline = settings.search_provider == "mock"
    if offline:
        st.info("Offline demo mode — mock search + built-in extraction, no API keys needed.")
    elif problems:
        st.warning("Configuration incomplete for *live* discovery:\n\n- " + "\n- ".join(problems))
    else:
        st.success("Configuration complete.")

    st.divider()
    st.markdown("### Run history")
    db = get_database()
    runs = db.list_runs(20)
    run_labels = {r["run_id"]: f"{r['run_id']}  ·  {r.get('company_count',0)} co  ·  {r.get('contact_count',0)} ct  ·  {r['status']}  ·  {fmt_date(r['started_at'])}"
                  for r in runs}
    run_ids = list(run_labels.keys())
    if run_ids:
        selected_run = st.selectbox("Show run", run_ids, format_func=lambda rid: run_labels.get(rid, rid))
    else:
        selected_run = None
        st.caption("No runs yet - upload company data and start discovery.")

    st.divider()
    st.markdown("### Services offered")
    for svc in IT_SERVICES:
        st.markdown(f"- {svc}")


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("B2B Decision-Maker Contact Finder")
st.caption("Receives companies from **Agent 1** (company research) → finds decision-makers and their publicly available professional contact details.")

st.info(
    "Discovery never sends emails or LinkedIn messages. "
    "Contact details are either **published in a source** (marked "
    f"`{EmailStatus.PUBLISHED.value}`) or **verified by a configured provider** "
    f"(`{EmailStatus.VERIFIED.value}`). Blank fields are unavailable - Agent 2 never invents them."
)

tab_upload, tab_discover, tab_browse, tab_review, tab_export = st.tabs(
    ["1 · Upload companies", "2 · Start discovery", "3 · Browse contacts", "4 · Review & approve", "5 · Export"]
)

# ===========================================================================
# Tab 1 - Upload
# ===========================================================================
with tab_upload:
    st.subheader("Upload company data from Agent 1")
    st.markdown(
        "Accept **JSON** (`{\"companies\": [...]}`, `{\"company\": {...}}`, or a raw list) or **CSV** "
        "with at least `company_name` and `company_id` columns."
    )
    uploaded = st.file_uploader("Choose a JSON or CSV file", type=["json", "csv"])
    if uploaded is not None:
        raw = uploaded.getvalue().decode("utf-8-sig", errors="replace")
        name_lower = uploaded.name.lower()
        if name_lower.endswith(".csv"):
            try:
                df = pd.read_csv(pd.io.common.StringIO(raw))
                result = load_companies_from_dataframe(df)
            except Exception as exc:
                st.error(f"Could not parse CSV: {exc}")
                result = None
        else:
            result = load_companies_from_json_text(raw)

        if result is not None and result.companies:
            st.session_state["uploaded_companies"] = result.companies
            st.session_state["uploaded_name"] = uploaded.name
            st.success(
                f"Loaded **{len(result.companies)}** companies from `{uploaded.name}`"
                + (f" — {len(result.errors)} records skipped" if result.errors else "")
            )
            preview = pd.DataFrame([{
                "company_id": c.company_id,
                "company_name": c.company_name,
                "website": c.website,
                "country": c.country,
                "industry": c.industry,
                "company_size": c.company_size,
            } for c in result.companies])
            st.dataframe(preview, use_container_width=True, hide_index=True)
        elif result is not None:
            st.warning("No valid companies found in that file." + (f" Errors: {'; '.join(result.errors[:5])}" if result.errors else ""))
        else:
            st.warning("Nothing loaded yet.")

    # also allow pasting JSON
    with st.expander("…or paste JSON"):
        pasted = st.text_area("Paste Agent 1 JSON output", height=200)
        if st.button("Load pasted JSON"):
            if pasted.strip():
                result = load_companies_from_json_text(pasted)
                if result.companies:
                    st.session_state["uploaded_companies"] = result.companies
                    st.success(f"Loaded **{len(result.companies)}** companies from pasted JSON.")
                else:
                    st.warning("No companies parsed." + (f" Errors: {'; '.join(result.errors[:5])}" if result.errors else ""))
            else:
                st.info("Paste JSON first.")

    if "uploaded_companies" in st.session_state:
        st.caption(f"Ready: **{len(st.session_state['uploaded_companies'])}** companies staged for discovery.")

# ===========================================================================
# Tab 2 - Start discovery
# ===========================================================================
with tab_discover:
    st.subheader("Start contact discovery")
    companies = st.session_state.get("uploaded_companies")
    if not companies:
        st.info("Upload company data in **Tab 1** first.")
    else:
        st.write(f"**{len(companies)}** companies will be processed.")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Companies", len(companies))
        with col2:
            st.metric("Contacts target", "2–5 / company")
        with col3:
            st.metric("Search provider", settings.search_provider)

        st.warning(
            "Live discovery calls the configured **search provider** and the **Claude API** and may take "
            "several minutes per company. Use `SEARCH_PROVIDER=mock` in `.env` for an offline demo."
        )

        if st.button("▶ Start discovery", type="primary"):
            if problems and not offline:
                st.error("Cannot start live discovery while configuration is incomplete:\n\n- " + "\n- ".join(problems))
            else:
                with st.spinner("Discovering decision-makers… this can take a while."):
                    run_id, error = start_discovery(companies, settings, db)
                if error:
                    st.error(f"Discovery failed: {error}")
                else:
                    st.session_state["run_id"] = run_id
                    st.success(f"Discovery run **{run_id}** completed.")
                    contacts = len(db.list_contacts(run_id))
                    st.write(f"{contacts} contacts found across {len(companies)} companies.")

# ===========================================================================
# Tab 3 - Browse
# ===========================================================================
with tab_browse:
    st.subheader("Contacts grouped by company")
    rows = contacts_dataframe(selected_run, None, None, None, None, None)

    if rows.empty:
        st.info("No contacts yet. Run discovery first.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            countries = sorted({str(x) for x in rows["company_country"].dropna().unique() if str(x).strip()})
            country_filter = st.selectbox("Country", ["All"] + countries, key="browse_country")
        with c2:
            titles = sorted({str(x) for x in rows["title"].dropna().unique() if str(x).strip()})
            title_filter = st.selectbox("Job title", ["All"] + titles, key="browse_title")
        with c3:
            statuses = ["All"] + sorted(rows["email_status"].dropna().unique().tolist())
            email_filter = st.selectbox("Email status", statuses, key="browse_email")
        with c4:
            states = ["All", "approved", "pending", "rejected"]
            state_filter = st.selectbox("Review state", states, key="browse_state")

        display = rows.copy()
        if country_filter != "All":
            display = display[display["company_country"] == country_filter]
        if title_filter != "All":
            display = display[display["title"] == title_filter]
        if email_filter != "All":
            display = display[display["email_status"] == email_filter]
        if state_filter != "All":
            display = display[display["review_state"] == state_filter]

        st.caption(f"{len(display)} of {len(rows)} contacts shown")
        for company_id, group in display.groupby("company_id"):
            company_name = group["company_name"].iloc[0]
            with st.expander(f"🏢 {company_name}  ({len(group)} contacts)", expanded=(len(display) <= 10)):
                for _, row in group.iterrows():
                    st.markdown(f"**{row['name'] or '—'}**  ·  {row['title'] or 'title n/a'}")
                    if row.get("department"):
                        st.markdown(f"Dept: {row['department']}")
                    if row.get("linkedin_url"):
                        st.markdown(f"[LinkedIn]({row['linkedin_url']})")
                    if row.get("work_email"):
                        st.markdown(f"✉ {row['work_email']}  ·  `{row['email_status']}`")
                    if row.get("business_phone"):
                        st.markdown(f"☎ {row['business_phone']}  ·  `{row['phone_status']}`")
                    if row.get("role_relevance"):
                        st.markdown(f"💡 {row['role_relevance']}")
                    if row.get("source_urls"):
                        urls = json.loads(row["source_urls"]) if isinstance(row["source_urls"], str) else row["source_urls"]
                        for u in urls[:3]:
                            st.markdown(f"🔗 [source]({u})")
                    if row.get("confidence_score"):
                        st.progress(float(row["confidence_score"]), text=f"confidence {float(row['confidence_score']):.0%}")
                    st.markdown("---")

# ===========================================================================
# Tab 4 - Review & approve
# ===========================================================================
with tab_review:
    st.subheader("Review, edit and approve contacts")
    rows = contacts_dataframe(selected_run, None, None, None, None, None)
    if rows.empty:
        st.info("No contacts to review yet.")
    else:
        status_legend = {
            "published_in_source_not_independently_verified": "🟡 published in source, not independently verified",
            "verified_by_configured_provider": "🟢 verified by configured provider",
            "invalid": "🔴 invalid",
            "catch_all_or_uncertain": "🟠 catch-all / uncertain",
            "not_found": "⚪ not found",
        }
        st.caption("Legend: " + " · ".join(status_legend.values()))

        pick_options = [f"{r['company_name']} · {r['name']} · {r['title']}" for _, r in rows.iterrows()]
        pick = st.selectbox("Select a contact to review", ["—"] + pick_options)
        if pick != "—":
            idx = pick_options.index(pick)
            row = rows.iloc[idx]
            company_id = row["company_id"]
            contact_key_val = row["contact_key"]
            run_for_row = selected_run or str(row.get("run_id", ""))

            st.markdown("#### Current values")
            col_l, col_r = st.columns(2)
            with col_l:
                st.write(f"**Name:** {row['name']}")
                st.write(f"**Title:** {row['title']}")
                st.write(f"**Dept:** {row['department']}")
                st.write(f"**Country:** {row['country']}")
                st.write(f"**LinkedIn:** {row['linkedin_url']}")
            with col_r:
                st.write(f"**Email:** {row['work_email']} · `{row['email_status']}`")
                st.write(f"**Phone:** {row['business_phone']} · `{row['phone_status']}`")
                st.write(f"**Confidence:** {float(row['confidence_score']):.0%}" if row.get('confidence_score') else "**Confidence:** —")
                st.write(f"**Review state:** {row['review_state']}")
                if row.get("verification_notes"):
                    st.write(f"**Notes:** {row['verification_notes']}")

            st.markdown("#### Edit")
            with st.form(f"edit_form_{contact_key_val}"):
                new_name = st.text_input("Name", value=row["name"] or "")
                new_title = st.text_input("Title", value=row["title"] or "")
                new_linkedin = st.text_input("LinkedIn URL", value=row["linkedin_url"] or "")
                new_email = st.text_input("Work email", value=row["work_email"] or "")
                new_email_status = st.selectbox(
                    "Email status", EmailStatus.values(),
                    index=EmailStatus.values().index(row["email_status"]) if row["email_status"] in EmailStatus.values() else 0,
                )
                new_phone = st.text_input("Business phone", value=row["business_phone"] or "")
                new_phone_status = st.selectbox(
                    "Phone status", PhoneStatus.values(),
                    index=PhoneStatus.values().index(row["phone_status"]) if row["phone_status"] in PhoneStatus.values() else 0,
                )
                new_reason = st.text_area("Role relevance", value=row["role_relevance"] or "")
                new_notes = st.text_area("Verification notes", value=row["verification_notes"] or "")

                col_b1, col_b2, col_b3 = st.columns(3)
                saved = col_b1.form_submit_button("💾 Save changes", type="secondary")
                approved = col_b2.form_submit_button("✅ Approve", type="primary")
                rejected = col_b3.form_submit_button("🚫 Reject")

            if approved or rejected or saved:
                fields = {
                    "name": new_name,
                    "title": new_title,
                    "linkedin_url": new_linkedin,
                    "work_email": new_email,
                    "email_status": new_email_status,
                    "business_phone": new_phone,
                    "phone_status": new_phone_status,
                    "role_relevance": new_reason,
                    "verification_notes": new_notes,
                }
                db.update_contact_fields(run_for_row, company_id, contact_key_val, fields)
                if approved:
                    db.set_review_state(run_for_row, company_id, contact_key_val, "approved")
                    st.toast("Contact approved ✅")
                elif rejected:
                    db.set_review_state(run_for_row, company_id, contact_key_val, "rejected")
                    st.toast("Contact rejected")
                else:
                    st.toast("Changes saved")
                st.rerun()

# ===========================================================================
# Tab 5 - Export
# ===========================================================================
with tab_export:
    st.subheader("Export contacts")
    rows = contacts_dataframe(selected_run, None, None, None, None, None)
    if rows.empty:
        st.info("Nothing to export yet - run discovery first.")
    else:
        st.caption(f"{len(rows)} contacts in the database for the current run.")
        include_rejected = st.checkbox("Include rejected contacts", value=False)
        fmt_json = st.checkbox("JSON", value=True)
        fmt_csv = st.checkbox("CSV", value=True)
        fmt_xlsx = st.checkbox("Excel", value=False)

        from agent2.models import CompanyContacts, Contact
        from agent2.exporters import export_csv, export_excel, export_json
        from pathlib import Path

        def rows_to_results(df: pd.DataFrame) -> list:
            groups = {}
            for _, r in df.iterrows():
                key = r["company_id"]
                groups.setdefault(key, {"company_id": key, "company_name": r.get("company_name", ""), "contacts": []})
                contact = Contact(
                    name=r.get("name", "") or "",
                    title=r.get("title", "") or "",
                    department=r.get("department", "") or "",
                    country=r.get("country", "") or "",
                    linkedin_url=r.get("linkedin_url", "") or "",
                    work_email=r.get("work_email", "") or "",
                    email_status=r.get("email_status", "not_found") or "not_found",
                    business_phone=r.get("business_phone", "") or "",
                    phone_status=r.get("phone_status", "not_found") or "not_found",
                    official_company_contact_page=r.get("official_company_contact_page", "") or "",
                    source_identity_url=r.get("source_identity_url", "") or "",
                    source_contact_url=r.get("source_contact_url", "") or "",
                    role_relevance=r.get("role_relevance", "") or "",
                    source_urls=json.loads(r["source_urls"]) if isinstance(r.get("source_urls"), str) and r.get("source_urls") else [],
                    confidence_score=float(r.get("confidence_score") or 0),
                    verification_notes=r.get("verification_notes", "") or "",
                    research_date=r.get("research_date", "") or "",
                    review_state=r.get("review_state", "pending") or "pending",
                    source_tags=json.loads(r["source_tags"]) if isinstance(r.get("source_tags"), str) and r.get("source_tags") else [],
                )
                groups[key]["contacts"].append(contact)
            return [CompanyContacts(company_id=g["company_id"], company_name=g["company_name"], contacts=g["contacts"])
                    for g in groups.values()]

        def default_filename(ext: str) -> str:
            stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M")
            return f"contacts_{selected_run or 'all'}_{stamp}.{ext}"

        if st.button("⬇ Export now", type="primary"):
            selected_rows = rows if include_rejected else rows[rows["review_state"] != "rejected"]
            if selected_rows.empty:
                st.warning("No contacts match the current filter.")
            else:
                results = rows_to_results(selected_rows)
                out = Path(settings.output_dir)
                out.mkdir(parents=True, exist_ok=True)
                written = []
                if fmt_json:
                    p = export_json(results, out / default_filename("json"))
                    written.append(p)
                if fmt_csv:
                    p = export_csv(results, out / default_filename("csv"))
                    written.append(p)
                if fmt_xlsx:
                    p = export_excel(results, out / default_filename("xlsx"))
                    written.append(p)
                for p in written:
                    st.success(f"Wrote {p}")

        st.divider()
        st.markdown("#### Export format (one row per contact)")
        st.code(", ".join(EXPORT_COLUMNS))


st.caption("Agent 2 · B2B Decision-Maker Contact Finder · evidence-backed, never invented.")