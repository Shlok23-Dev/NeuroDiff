"""
Neuro-Diff — Context-Aware AI Teaching Assistant & Visual Misconception Engine
Gemma 4 Hackathon — Next-Gen AI Education Track

Single-file Streamlit application.
"""

import os
import json
import traceback
from enum import Enum
from typing import List, Optional

import streamlit as st

# ------------------------------------------------------------------
# Streamlit page config MUST be the very first Streamlit command.
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Neuro-Diff",
    layout="wide",
    initial_sidebar_state="collapsed",
)

from pydantic import BaseModel, Field
from PIL import Image
import PyPDF2

from streamlit_agraph import agraph, Node as AGNode, Edge as AGEdge, Config as AGConfig

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None


# ====================================================================
# 1. CONSTANTS / THEME
# ====================================================================

MODEL_NAME = "gemma-4-12b-unified"

COLOR_BG = "#0B0F19"
COLOR_SURFACE = "#131A29"
COLOR_TEXT = "#F8FAFC"
COLOR_BORDER = "#1E293B"
COLOR_ACCENT = "#FF4B4B"

COLOR_VALID = "#00FFA3"      # Neon Mint
COLOR_MISSING = "#334155"    # Slate Gray
COLOR_COLLISION = "#FF3366"  # Neon Crimson

MAX_SYLLABUS_CHARS = 15000


# ====================================================================
# 2. PYDANTIC SCHEMAS (Structured Outputs — no hallucinations)
# ====================================================================

class NodeStatus(str, Enum):
    VALID = "VALID"
    MISSING = "MISSING"
    COLLISION = "COLLISION"


class Node(BaseModel):
    id: str = Field(description="Unique short identifier for this concept node")
    label: str = Field(description="Human readable concept name")
    status: NodeStatus = Field(description="VALID, MISSING, or COLLISION")


class Edge(BaseModel):
    source: str = Field(description="Source node id")
    target: str = Field(description="Target node id")
    status: NodeStatus = Field(description="VALID, MISSING, or COLLISION")


class CognitiveGraph(BaseModel):
    nodes: List[Node]
    edges: List[Edge]
    cognitive_diagnosis: str = Field(
        description="A narrative paragraph diagnosing the exact point of "
        "conceptual divergence in the student's mental model."
    )


class QuestionSet(BaseModel):
    questions: List[str] = Field(description="List of grounded exam questions")


# ====================================================================
# 3. CUSTOM CSS — Deep Obsidian Dark Mode
# ====================================================================

def inject_custom_css():
    st.markdown(
        f"""
        <style>
            #MainMenu {{visibility: hidden;}}
            footer {{visibility: hidden;}}
            header {{visibility: hidden;}}

            .stApp {{
                background-color: {COLOR_BG};
                color: {COLOR_TEXT};
            }}

            .block-container {{
                padding-top: 2rem;
                padding-bottom: 2rem;
            }}

            .neuro-card {{
                background-color: {COLOR_SURFACE};
                border: 1px solid {COLOR_BORDER};
                border-radius: 12px;
                padding: 1.25rem 1.5rem;
                margin-bottom: 1rem;
            }}

            .neuro-title {{
                font-size: 2.2rem;
                font-weight: 800;
                color: {COLOR_TEXT};
                margin-bottom: 0;
            }}

            .neuro-subtitle {{
                color: #94A3B8;
                font-size: 1rem;
                margin-top: 0;
                margin-bottom: 1.5rem;
            }}

            .status-pill {{
                display: inline-block;
                padding: 0.2rem 0.7rem;
                border-radius: 999px;
                font-size: 0.75rem;
                font-weight: 700;
                letter-spacing: 0.03em;
                margin-right: 0.4rem;
            }}

            .pill-valid {{ background-color: rgba(0,255,163,0.15); color: {COLOR_VALID}; border: 1px solid {COLOR_VALID}; }}
            .pill-missing {{ background-color: rgba(51,65,85,0.4); color: #94A3B8; border: 1px solid {COLOR_MISSING}; }}
            .pill-collision {{ background-color: rgba(255,51,102,0.15); color: {COLOR_COLLISION}; border: 1px solid {COLOR_COLLISION}; }}

            div.stButton > button {{
                background-color: {COLOR_ACCENT};
                color: white;
                border: none;
                border-radius: 8px;
                padding: 0.5rem 1.2rem;
                font-weight: 600;
            }}
            div.stButton > button:hover {{
                background-color: #E03E3E;
                color: white;
            }}

            section[data-testid="stFileUploader"] {{
                background-color: {COLOR_SURFACE};
                border: 1px dashed {COLOR_BORDER};
                border-radius: 12px;
                padding: 1rem;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ====================================================================
# 4. SESSION STATE INITIALIZATION
# ====================================================================

def init_session_state():
    defaults = {
        "syllabus_text": None,
        "syllabus_filename": None,
        "questions": None,
        "selected_question": None,
        "cognitive_graph": None,
        "graph_source": None,  # "live" or "fallback"
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ====================================================================
# 5. API CLIENT / KEY SAFETY
# ====================================================================

@st.cache_resource(show_spinner=False)
def get_client():
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        st.error(
            "Missing API key. Please set the `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) "
            "environment variable before launching Neuro-Diff."
        )
        st.stop()
    if genai is None:
        st.error(
            "The `google-genai` package is not installed. Run `pip install google-genai`."
        )
        st.stop()
    try:
        client = genai.Client(api_key=api_key)
        return client
    except Exception as e:
        st.error(f"Failed to initialize the Gemma client: {e}")
        st.stop()


# ====================================================================
# 6. RAG — PDF EXTRACTION (with context safety valve)
# ====================================================================

def extract_pdf_text(uploaded_file) -> Optional[str]:
    """Extract text from an uploaded PDF, truncated to MAX_SYLLABUS_CHARS."""
    try:
        reader = PyPDF2.PdfReader(uploaded_file)
        if len(reader.pages) == 0:
            st.error("The uploaded PDF appears to have no pages.")
            return None

        full_text = ""
        for page in reader.pages:
            try:
                page_text = page.extract_text() or ""
            except Exception:
                page_text = ""
            full_text += page_text + "\n"

        full_text = full_text.strip()
        if not full_text:
            st.error(
                "No extractable text was found in this PDF. It may be a scanned "
                "image-only document."
            )
            return None

        truncated = full_text[:MAX_SYLLABUS_CHARS]
        return truncated

    except PyPDF2.errors.PdfReadError:
        st.error("This file appears to be corrupted or is not a valid PDF.")
        return None
    except Exception as e:
        st.error(f"Unexpected error while parsing the PDF: {e}")
        return None


# ====================================================================
# 7. QUESTION GENERATION (Gemma 4, structured output)
# ====================================================================

def generate_questions(client, syllabus_text: str, num_questions: int = 5) -> Optional[List[str]]:
    prompt = f"""You are an expert exam-setter. Based strictly on the following syllabus
excerpt, generate {num_questions} grounded, conceptually rich exam questions that
would reveal a student's underlying understanding (or misconceptions) of the
material. Do not invent topics not present in the syllabus.

SYLLABUS EXCERPT:
---
{syllabus_text}
---

Return only the questions.
"""
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=QuestionSet,
            ),
        )
        parsed: QuestionSet = response.parsed
        if parsed is None:
            parsed = QuestionSet.model_validate_json(response.text)
        if not parsed.questions:
            return None
        return parsed.questions
    except Exception as e:
        st.error(f"Question generation failed: {e}")
        return None


# ====================================================================
# 8. FALLBACK COGNITIVE GRAPH (Stage Safety Net)
# ====================================================================

def get_fallback_graph() -> CognitiveGraph:
    return CognitiveGraph(
        nodes=[
            Node(id="n1", label="Newton's Second Law", status=NodeStatus.VALID),
            Node(id="n2", label="Force = Mass x Acceleration", status=NodeStatus.VALID),
            Node(id="n3", label="Friction as Opposing Force", status=NodeStatus.COLLISION),
            Node(id="n4", label="Net Force Concept", status=NodeStatus.MISSING),
            Node(id="n5", label="Free Body Diagrams", status=NodeStatus.MISSING),
        ],
        edges=[
            Edge(source="n1", target="n2", status=NodeStatus.VALID),
            Edge(source="n2", target="n3", status=NodeStatus.COLLISION),
            Edge(source="n2", target="n4", status=NodeStatus.MISSING),
            Edge(source="n4", target="n5", status=NodeStatus.MISSING),
        ],
        cognitive_diagnosis=(
            "The student correctly recalls the formula F = ma but exhibits a "
            "collision at the point where friction is introduced as an opposing "
            "force — they appear to treat friction as additive rather than "
            "subtractive to net force. The concept of 'net force' as an "
            "aggregation step is entirely missing from their reasoning chain, "
            "which cascades into an absent understanding of free body diagrams. "
            "(This is cached fallback data — live telemetry was unavailable.)"
        ),
    )


# ====================================================================
# 9. MULTIMODAL EVALUATION (Gemma 4, image + text -> structured output)
# ====================================================================

def evaluate_student_answer(
    client, image: Image.Image, syllabus_context: str, question: str
) -> CognitiveGraph:
    """
    Evaluate a student's handwritten answer image against the syllabus context.
    Wrapped defensively — on any failure, falls back to a cached, schema-valid
    CognitiveGraph so the app never crashes on stage.
    """
    prompt = f"""You are a cognitive diagnostician analyzing a student's handwritten
answer. Reason directly from the attached image.

SYLLABUS CONTEXT (ground truth):
---
{syllabus_context}
---

EXAM QUESTION:
{question}

Build a Cognitive Diff Graph representing the student's mental model:
- Each node is a concept the student engaged with (or should have engaged with).
- status VALID: the student demonstrated correct understanding of this concept.
- status MISSING: an expected concept is entirely absent from the student's reasoning.
- status COLLISION: the student's reasoning directly conflicts with the correct concept
  (a genuine misconception).
- Each edge represents a logical/causal link between two concepts, with its own status.

Also produce a narrative `cognitive_diagnosis`: a clear, specific paragraph
pinpointing the exact moment/concept where the student's reasoning diverges
from the correct mental model.
"""
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[prompt, image],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=CognitiveGraph,
            ),
        )
        parsed: CognitiveGraph = response.parsed
        if parsed is None:
            parsed = CognitiveGraph.model_validate_json(response.text)
        if not parsed.nodes:
            raise ValueError("Model returned an empty node list.")
        st.session_state["graph_source"] = "live"
        return parsed

    except Exception:
        # Stage safety net — never crash, always show something realistic.
        st.toast(
            "API Telemetry Offline — Switching to Cached Graph Data", icon="⚠️"
        )
        st.session_state["graph_source"] = "fallback"
        return get_fallback_graph()


# ====================================================================
# 10. GRAPH RENDERING (streamlit-agraph)
# ====================================================================

STATUS_COLOR_MAP = {
    NodeStatus.VALID: COLOR_VALID,
    NodeStatus.MISSING: COLOR_MISSING,
    NodeStatus.COLLISION: COLOR_COLLISION,
}


def render_cognitive_graph(graph: CognitiveGraph):
    if not graph or not graph.nodes:
        st.warning("No graph data available to render.")
        return

    ag_nodes = []
    for n in graph.nodes:
        color = STATUS_COLOR_MAP.get(n.status, COLOR_MISSING)
        ag_nodes.append(
            AGNode(
                id=n.id,
                label=n.label,
                size=22,
                color=color,
                font={"color": COLOR_TEXT, "size": 14},
                shape="dot",
            )
        )

    ag_edges = []
    for e in graph.edges:
        color = STATUS_COLOR_MAP.get(e.status, COLOR_MISSING)
        dashed = e.status == NodeStatus.MISSING
        width = 4 if e.status == NodeStatus.COLLISION else 2
        ag_edges.append(
            AGEdge(
                source=e.source,
                target=e.target,
                color=color,
                width=width,
                dashes=dashed,
            )
        )

    config = AGConfig(
        width="100%",
        height=560,
        directed=True,
        physics=False,
        hierarchical=True,
        layout={
            "hierarchical": {
                "enabled": True,
                "direction": "UD",
                "sortMethod": "directed",
                "levelSeparation": 110,
                "nodeSpacing": 140,
            }
        },
        nodeHighlightBehavior=True,
        highlightColor=COLOR_ACCENT,
        collapsible=False,
        node={"labelProperty": "label"},
        link={"renderLabel": False},
    )

    try:
        agraph(nodes=ag_nodes, edges=ag_edges, config=config)
    except Exception as e:
        st.error(f"The Cognitive Diff Graph failed to render: {e}")


def render_legend():
    st.markdown(
        f"""
        <div>
            <span class="status-pill pill-valid">VALID</span>
            <span class="status-pill pill-missing">MISSING</span>
            <span class="status-pill pill-collision">COLLISION</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ====================================================================
# 11. UI — MAIN APP FLOW
# ====================================================================

def render_header():
    st.markdown('<p class="neuro-title">🧠 Neuro-Diff</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="neuro-subtitle">Context-Aware AI Teaching Assistant &amp; '
        "Visual Misconception Engine — powered by Gemma 4</p>",
        unsafe_allow_html=True,
    )


def stage_1_syllabus_upload():
    st.markdown('<div class="neuro-card">', unsafe_allow_html=True)
    st.subheader("Stage 1 — Ingest Syllabus (RAG)")

    uploaded_pdf = st.file_uploader(
        "Upload a syllabus PDF", type=["pdf"], key="syllabus_uploader"
    )

    if uploaded_pdf is not None:
        # Only re-parse if it's a new file (avoid re-running API/parsing on every rerun)
        if st.session_state["syllabus_filename"] != uploaded_pdf.name:
            with st.spinner("Extracting syllabus text..."):
                text = extract_pdf_text(uploaded_pdf)
            if text:
                st.session_state["syllabus_text"] = text
                st.session_state["syllabus_filename"] = uploaded_pdf.name
                st.session_state["questions"] = None
                st.session_state["selected_question"] = None
                st.session_state["cognitive_graph"] = None
                st.success(f"Parsed `{uploaded_pdf.name}` successfully.")

    if st.session_state["syllabus_text"]:
        with st.expander("Preview extracted syllabus text"):
            st.text(st.session_state["syllabus_text"][:2000] + " ...")

    st.markdown("</div>", unsafe_allow_html=True)


def stage_2_question_generation():
    if not st.session_state["syllabus_text"]:
        return

    st.markdown('<div class="neuro-card">', unsafe_allow_html=True)
    st.subheader("Stage 2 — Generate Grounded Exam Questions")

    col1, col2 = st.columns([1, 3])
    with col1:
        generate_clicked = st.button("Generate Questions", key="gen_questions_btn")

    if generate_clicked:
        client = get_client()
        with st.spinner("Gemma 4 is drafting grounded questions..."):
            questions = generate_questions(client, st.session_state["syllabus_text"])
        if questions:
            st.session_state["questions"] = questions
            st.session_state["selected_question"] = questions[0]
            st.toast("Questions generated.", icon="✅")
        else:
            st.error("Question generation returned no usable output.")

    if st.session_state["questions"]:
        st.session_state["selected_question"] = st.radio(
            "Select a question to evaluate against:",
            options=st.session_state["questions"],
            key="question_radio",
        )

    st.markdown("</div>", unsafe_allow_html=True)


def stage_3_student_answer():
    if not st.session_state["questions"]:
        return

    st.markdown('<div class="neuro-card">', unsafe_allow_html=True)
    st.subheader("Stage 3 — Upload Student's Handwritten Answer")

    uploaded_image = st.file_uploader(
        "Upload an image of the student's answer",
        type=["png", "jpg", "jpeg"],
        key="student_answer_uploader",
    )

    evaluate_clicked = False
    image_obj = None
    if uploaded_image is not None:
        try:
            image_obj = Image.open(uploaded_image)
            st.image(image_obj, caption="Student Answer", use_container_width=True)
        except Exception as e:
            st.error(f"Could not open this image: {e}")
            image_obj = None

        if image_obj is not None:
            evaluate_clicked = st.button("Diagnose Cognitive Model", key="evaluate_btn")

    if evaluate_clicked and image_obj is not None:
        client = get_client()
        with st.spinner("Gemma 4 is reasoning over the handwritten answer..."):
            try:
                graph = evaluate_student_answer(
                    client,
                    image_obj,
                    st.session_state["syllabus_text"],
                    st.session_state["selected_question"],
                )
            except Exception:
                st.toast(
                    "API Telemetry Offline — Switching to Cached Graph Data",
                    icon="⚠️",
                )
                st.session_state["graph_source"] = "fallback"
                graph = get_fallback_graph()
        st.session_state["cognitive_graph"] = graph

    st.markdown("</div>", unsafe_allow_html=True)


def stage_4_diagnosis():
    graph: Optional[CognitiveGraph] = st.session_state["cognitive_graph"]
    if not graph:
        return

    st.markdown('<div class="neuro-card">', unsafe_allow_html=True)
    st.subheader("Stage 4 — Cognitive Diagnosis & Diff Graph")

    if st.session_state.get("graph_source") == "fallback":
        st.warning(
            "Displaying cached fallback data — live model telemetry was unavailable.",
            icon="⚠️",
        )

    render_legend()

    left, right = st.columns([1, 1])
    with left:
        st.markdown("##### Cognitive Diff Graph")
        if not graph.nodes:
            st.info("The model returned an empty graph — nothing to visualize yet.")
        else:
            render_cognitive_graph(graph)

    with right:
        st.markdown("##### Narrative Diagnosis")
        st.write(graph.cognitive_diagnosis)

        st.markdown("##### Node Status Breakdown")
        for n in graph.nodes:
            pill_class = {
                NodeStatus.VALID: "pill-valid",
                NodeStatus.MISSING: "pill-missing",
                NodeStatus.COLLISION: "pill-collision",
            }.get(n.status, "pill-missing")
            st.markdown(
                f'<span class="status-pill {pill_class}">{n.status.value}</span> {n.label}',
                unsafe_allow_html=True,
            )

    st.markdown("</div>", unsafe_allow_html=True)


# ====================================================================
# 12. MAIN
# ====================================================================

def main():
    inject_custom_css()
    init_session_state()
    render_header()

    stage_1_syllabus_upload()
    stage_2_question_generation()
    stage_3_student_answer()
    stage_4_diagnosis()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error("Neuro-Diff hit an unexpected error and halted safely.")
        st.exception(e)
