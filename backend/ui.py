import requests
import streamlit as st

API_BASE = "http://127.0.0.1:8000"

st.set_page_config(page_title="RAG Research QA", layout="wide")
st.title("RAG Research QA (Doc-filtered)")

# ---- Load documents ----


@st.cache_data(ttl=10)
def fetch_documents():
    r = requests.get(f"{API_BASE}/documents", timeout=10)
    r.raise_for_status()
    return r.json()["items"]


if st.button("Refresh documents list"):
    fetch_documents.clear()

docs = fetch_documents()

st.subheader("Ingest a new document")

uploaded = st.file_uploader(
    "Upload PDF/MD/TXT", type=["pdf", "md", "markdown", "txt"])
new_title = st.text_input("Title (optional)", value="")
replace = st.checkbox("Replace if same file uploaded", value=True)

if st.button("Ingest"):
    if not uploaded:
        st.warning("Please choose a file first.")
    else:
        files = {"file": (uploaded.name, uploaded.getvalue(),
                          uploaded.type or "application/octet-stream")}
        data = {"title": new_title, "replace": str(replace).lower()}
        r = requests.post(f"{API_BASE}/ingest",
                          files=files, data=data, timeout=300)
        if r.status_code != 200:
            st.error(f"/ingest failed: HTTP {r.status_code}")
            st.code(r.text[:4000])
        else:
            st.success("Ingested successfully!")
            st.json(r.json())
            # refresh list
            fetch_documents.clear()
            st.rerun()

if not docs:
    st.warning("No documents found. Ingest a PDF/MD first.")
    st.stop()

# Dropdown display: "Title (filename) — id"


def label(d):
    fn = d.get("filename") or "unknown"
    return f"{d['title']} ({fn}) — id={d['id']}"


doc_map = {label(d): d for d in docs}
choice = st.selectbox("Select a document", list(doc_map.keys()))
doc = doc_map[choice]
document_id = doc["id"]

st.caption(f"Selected document_id = {document_id}")

# ---- Ask form ----
with st.form("ask_form"):
    query = st.text_area(
        "Question", value="What is the main topic of this PDF?", height=120)
    top_k = st.slider("top_k (retrieval)", min_value=1, max_value=10, value=4)
    submitted = st.form_submit_button("Ask")

if submitted:
    payload = {"query_text": query, "top_k": top_k, "document_id": document_id}
    r = requests.post(f"{API_BASE}/ask", json=payload, timeout=60)
    if r.status_code != 200:
        st.error(f"/ask failed: HTTP {r.status_code}\n\n{r.text}")
        st.stop()

    data = r.json()

    # ---- Answer ----
    st.subheader("Answer")
    st.write(data.get("answer", ""))

    # ---- Citations ----
    st.subheader("Citations (used)")
    citations = data.get("citations", [])
    if not citations:
        st.info("No citations. (May have refused or fallback behavior.)")
    else:
        for c in citations:
            page = c.get("page")
            section = c.get("section")
            level = c.get("level")
            fn = c.get("filename") or c.get("filename_meta") or "unknown"
            src_type = c.get("source_type")

            where = []
            if page is not None:
                where.append(f"p.{page}")
            if section:
                where.append(
                    f"section: {section} (H{level})" if level else f"section: {section}")
            if src_type:
                where.append(src_type)

            header = f"[{c['chunk_id']}] {fn} — " + \
                ", ".join(where) if where else f"[{c['chunk_id']}] {fn}"
            st.markdown(f"**{header}**")
            st.code(c.get("snippet", ""), language="text")

    # ---- Debug panels ----
    with st.expander("Debug: threshold / selected / retrieved"):
        st.json({
            "threshold": data.get("threshold"),
            "cited_chunk_ids": data.get("cited_chunk_ids"),
            "refused": data.get("refused"),
            "reason": data.get("reason"),
        })
        st.markdown("**Selected Chunks**")
        st.json(data.get("selected_chunks", []))
        st.markdown("**Retrieved Chunks**")
        st.json(data.get("retrieved_chunks", []))

    # ---- Citation text ----
    citation_text = data.get("citation_text")
    if citation_text:
        st.subheader("Citation Text (copy/paste)")
        st.text_area("citation_text", value=citation_text, height=180)
