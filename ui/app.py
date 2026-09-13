"""Streamlit demo (FR-U1..U4). Talks to the API over HTTP only — it imports nothing from `rag`."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import altair as alt
import httpx
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # streamlit puts ui/ on the path, not the repo root

from ui.view import api_url, citation_panels, load_presets, parse_sse, waterfall_rows  # noqa: E402

API = api_url()
PRESETS = dict(load_presets(ROOT / "eval" / "adversarial" / "suite.yaml"))
VERDICT_COLOURS = {
    "pass": "#2e7d32",
    "redact": "#1565c0",
    "hedge": "#ef6c00",
    "refuse": "#c62828",
    "block": "#b71c1c",
    "skipped": "#9e9e9e",
    "error": "#6a1b9a",
}

st.set_page_config(page_title="Enterprise RAG", page_icon="🛡️", layout="wide")


def sidebar() -> tuple[bool, str | None]:
    with st.sidebar:
        st.header("Enterprise RAG")
        st.caption("CPU-only · zero marginal cost · tiered guardrails")
        try:
            health = httpx.get(f"{API}/health", timeout=5).json()
        except httpx.HTTPError:
            health = {"status": "unreachable", "dependencies": {}}
        st.metric("API", health["status"])
        for name, ready in health.get("dependencies", {}).items():
            st.caption(f"{'✅' if ready else '❌'} {name}")
        rerank = st.toggle(
            "Rerank", value=False, help="Off by default: measured negative lift (ADR-003)."
        )
        choice = st.selectbox("Trip a rail with a preset", ["—", *PRESETS])
        st.caption(
            "Generation runs on CPU: expect 10-60 s per answer. Tokens stream as they arrive."
        )
    return rerank, None if choice == "—" else PRESETS[choice]


def render_trace(trace: dict[str, Any]) -> None:
    rows = waterfall_rows(trace)
    if not rows:
        return
    verdict = trace.get("final_verdict", "pass")
    escalated = " · escalated to T3" if trace.get("escalated") else ""
    st.markdown(
        f"**Guardrail trace** — final verdict `{verdict}` · "
        f"{trace.get('total_latency_ms', 0):.0f} ms of rails{escalated}"
    )
    chart = (
        alt.Chart(alt.Data(values=[{k: v for k, v in r.items() if k != "evidence"} for r in rows]))
        .mark_bar()
        .encode(
            x=alt.X("start_ms:Q", title="rail time (ms) — generation excluded"),
            x2="end_ms:Q",
            y=alt.Y("rail:N", sort=None, title=None),
            color=alt.Color(
                "verdict:N",
                scale=alt.Scale(domain=list(VERDICT_COLOURS), range=list(VERDICT_COLOURS.values())),
            ),
            tooltip=["stage:N", "tier:N", "rail:N", "verdict:N", "score:Q", "latency_ms:Q"],
        )
    )
    st.altair_chart(chart, use_container_width=True)
    for row in rows:
        score = "—" if row["score"] is None else f"{row['score']:.3f}"
        label = (
            f"{row['tier']} · {row['rail']} → {row['verdict']} "
            f"(score {score}, {row['latency_ms']:.0f} ms)"
        )
        with st.expander(label):
            st.json(row["evidence"])


def render_answer(answer: dict[str, Any]) -> None:
    for panel in citation_panels(answer):
        with st.expander(panel["title"]):
            st.caption(panel["location"])
            st.code(panel["text"], language=panel["language"])
    if answer.get("stripped_markers"):
        st.caption(f"Fabricated citation markers stripped: {', '.join(answer['stripped_markers'])}")
    render_trace(answer.get("trace") or {})


def ask(query: str, rerank: bool) -> dict[str, Any] | None:
    final: dict[str, Any] = {}

    def tokens() -> Iterator[str]:
        body = {"query": query, "rerank": rerank, "include_trace": True, "stream": True}
        with httpx.stream("POST", f"{API}/query", json=body, timeout=300) as response:
            response.raise_for_status()
            for event in parse_sse(response.iter_lines()):
                if event["type"] == "token":
                    yield event["text"]
                elif event["type"] == "final":
                    final.update(event["answer"])

    streamed = st.write_stream(tokens())
    if final and final.get("text") != streamed:
        # Output rails run on the completed text; the terminal event is authoritative (FR-G5).
        st.info("The output rails changed the streamed answer. Final answer:")
        st.markdown(final["text"])
    return final or None


rerank, preset = sidebar()
st.session_state.setdefault("history", [])

for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["text"])
        if turn.get("answer"):
            render_answer(turn["answer"])

prompt = st.chat_input("Ask about this repository…")
if preset and st.session_state.get("last_preset") != preset:
    st.session_state.last_preset = preset
    prompt = preset

if prompt:
    st.session_state.history.append({"role": "user", "text": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        answer: dict[str, Any] | None
        try:
            answer = ask(prompt, rerank)
        except httpx.HTTPError as exc:
            st.error(f"API error: {exc}")
            answer = None
        if answer:
            render_answer(answer)
            st.session_state.history.append(
                {"role": "assistant", "text": answer["text"], "answer": answer}
            )
