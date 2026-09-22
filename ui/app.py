"""
Streamlit UI for the AI Research Paper Intelligence & Multimodal Analysis System.

Run with:
    streamlit run ui/app.py

Talks to the FastAPI backend over HTTP so the UI stays fully decoupled from
business logic.
"""

from __future__ import annotations

import os
import sys

import requests
import streamlit as st

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

API_BASE = os.environ.get("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="Research Assistant", page_icon="📄", layout="wide")


def _api_get(path: str):
    return requests.get(f"{API_BASE}{path}", timeout=60)


def _api_post(path: str, **kwargs):
    return requests.post(f"{API_BASE}{path}", timeout=180, **kwargs)


def _api_delete(path: str):
    return requests.delete(f"{API_BASE}{path}", timeout=60)


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of dicts: {query, response}


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #

with st.sidebar:
    st.header("📚 Documents")

    uploaded_files = st.file_uploader(
        "Upload research papers (PDF)", type=["pdf"], accept_multiple_files=True
    )

    if uploaded_files and st.button("Process papers", use_container_width=True):
        progress = st.progress(0.0, text="Processing...")
        for i, uploaded_file in enumerate(uploaded_files):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                resp = _api_post("/documents/upload", files=files)
                if resp.status_code == 200:
                    st.success(f"✓ {uploaded_file.name} processed")
                else:
                    st.error(f"✗ {uploaded_file.name}: {resp.json().get('detail', resp.text)}")
            except requests.RequestException as exc:
                st.error(f"✗ {uploaded_file.name}: could not reach backend ({exc})")
            progress.progress((i + 1) / len(uploaded_files))
        progress.empty()

    st.divider()
    st.subheader("Uploaded papers")

    try:
        docs_resp = _api_get("/documents")
        docs = docs_resp.json() if docs_resp.status_code == 200 else []
    except requests.RequestException:
        docs = []
        st.warning("Backend not reachable. Is `uvicorn app.main:app` running?")

    if docs:
        for doc in docs:
            col1, col2 = st.columns([4, 1])
            col1.markdown(f"✓ **{doc.get('filename', 'unknown')}**")
            if col2.button("🗑", key=f"del_{doc['document_id']}"):
                _api_delete(f"/documents/{doc['document_id']}")
                st.rerun()
    else:
        st.caption("No papers uploaded yet.")

    st.divider()
    st.subheader("Vector DB")
    st.caption("✓ Ready" if docs else "○ Empty")

    if st.button("Clear vector database", type="secondary", use_container_width=True):
        _api_post("/documents/clear")
        st.session_state.chat_history = []
        st.rerun()

    st.divider()
    show_trace = st.checkbox("Show agent execution trace", value=True)


# --------------------------------------------------------------------------- #
# Main area
# --------------------------------------------------------------------------- #

st.title("🔬 Research Assistant")
st.caption("Multi-agent RAG system for analyzing and comparing research papers.")

if not docs:
    st.info("Upload one or more PDF research papers from the sidebar to get started.")

for turn in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(turn["query"])
    with st.chat_message("assistant"):
        st.markdown(turn["answer"])
        if turn.get("citations"):
            with st.expander("📎 Citations"):
                for c in turn["citations"]:
                    st.markdown(f"- **{c['filename']}**, page {c['page_number']} ({c.get('section') or 'N/A'})")
        if show_trace and turn.get("trace"):
            with st.expander("🧭 Agent execution trace"):
                for step in turn["trace"]:
                    st.markdown(f"- {step}")
                if turn.get("tools_called"):
                    st.markdown(f"**Tools called:** {', '.join(turn['tools_called'])}")
                st.markdown(f"**Review status:** {turn.get('review_status', 'unknown')}")

user_question = st.chat_input("Ask a question about the uploaded papers...")

if user_question:
    st.session_state.chat_history.append({"query": user_question, "answer": "", "citations": [], "trace": []})
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        with st.spinner("Running planner → retriever → analysis → review..."):
            try:
                resp = _api_post("/query", json={"query": user_question, "document_ids": []})
                if resp.status_code == 200:
                    data = resp.json()
                    st.markdown(data["answer"])
                    if data.get("citations"):
                        with st.expander("📎 Citations"):
                            for c in data["citations"]:
                                st.markdown(f"- **{c['filename']}**, page {c['page_number']} ({c.get('section') or 'N/A'})")
                    if show_trace:
                        with st.expander("🧭 Agent execution trace"):
                            for step in data.get("agent_trace", []):
                                st.markdown(f"- {step}")
                            if data.get("tools_called"):
                                st.markdown(f"**Tools called:** {', '.join(data['tools_called'])}")
                            st.markdown(f"**Review status:** {data.get('review_status')}")
                    st.session_state.chat_history[-1].update(
                        answer=data["answer"],
                        citations=data.get("citations", []),
                        trace=data.get("agent_trace", []),
                        tools_called=data.get("tools_called", []),
                        review_status=data.get("review_status"),
                    )
                else:
                    error_msg = f"Error: {resp.json().get('detail', resp.text)}"
                    st.error(error_msg)
                    st.session_state.chat_history[-1]["answer"] = error_msg
            except requests.RequestException as exc:
                error_msg = f"Could not reach backend: {exc}"
                st.error(error_msg)
                st.session_state.chat_history[-1]["answer"] = error_msg
