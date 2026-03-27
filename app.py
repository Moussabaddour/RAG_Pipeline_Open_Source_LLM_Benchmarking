import streamlit as st
import os
import pandas as pd
import time
import gc

from rag_pipeline import (
    split_documents,
    create_vectorstore,
    build_rag_chain
)
from llm_loader import load_llm
from dataset import DATASET
from evaluation import (
    answer_quality,
    faithfulness,
    context_utilization
)
from langchain_community.document_loaders import PyPDFLoader, TextLoader

# =========================
# CONFIG
# =========================

st.set_page_config(page_title="RAG Chat App", layout="wide")
st.title("📚 RAG Chat + Evaluation Dashboard")

# =========================
# SESSION STATE
# =========================

if "rag_pipelines" not in st.session_state:
    st.session_state.rag_pipelines = {}
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Input management (never touch these AFTER the widget renders) ---
# input_text  : the value we WANT the box to show on next render
# input_ver   : incrementing this key forces Streamlit to create a
#               brand-new widget, which picks up the new `value=` arg.
#               This sidesteps the "cannot modify after instantiation" error.
if "input_text" not in st.session_state:
    st.session_state.input_text = ""
if "input_ver" not in st.session_state:
    st.session_state.input_ver = 0

# =========================
# FILE MAPPING
# =========================

file_mapping = {
    "EigenGAN": "Article1.pdf",
    "MFGAN":    "Article2.pdf",
    "CMGAN":    "Article3.pdf",
    "Ontology Depth": "Article4.pdf"
}

# =========================
# SIDEBAR
# =========================

st.sidebar.header("📄 Select Paper")
paper_names = [p["paper"] for p in DATASET]
selected_paper = st.sidebar.selectbox("Choose paper", paper_names)
file_name = file_mapping.get(selected_paper)
file_path = os.path.join("data", file_name)
st.sidebar.caption(f"📄 File: {file_name}")

st.sidebar.header("🤖 Select Models")
models = ["mistral", "tinyllama", "gemma","qwen","phi3"]
selected_models = st.sidebar.multiselect("Choose models", models, default=["mistral"])

def load_single_document(fp):
    loader = PyPDFLoader(fp) if fp.endswith(".pdf") else TextLoader(fp)
    return loader.load()

if st.sidebar.button("📥 Load Selected Paper and Models"):
    if not file_name or not os.path.exists(file_path):
        st.sidebar.error(f"File not found: {file_path}")
        st.stop()

    st.sidebar.info("📄 Loading document...")
    docs = load_single_document(file_path)
    st.sidebar.info("✂️ Splitting...")
    chunks = split_documents(docs)
    st.sidebar.info("🔍 Creating vector store...")
    vectorstore = create_vectorstore(chunks)

    # Free old LLM native memory before allocating new contexts
    if st.session_state.rag_pipelines:
        for old_rag in st.session_state.rag_pipelines.values():
            try:
                steps = getattr(old_rag, "steps", [])
                llm_obj = steps[-1] if steps else None
                if hasattr(llm_obj, "client"):
                    llm_obj.client.close()
            except Exception:
                pass
        st.session_state.rag_pipelines = {}
        gc.collect()

    st.session_state.vectorstore = vectorstore
    st.session_state.chat_history = []
    st.session_state.input_text = ""
    st.session_state.input_ver += 1   # force fresh widget

    for model_name in selected_models:
        st.sidebar.info(f"🤖 Loading {model_name}...")
        llm = load_llm(model_name)
        rag = build_rag_chain(llm, vectorstore)
        st.session_state.rag_pipelines[model_name] = rag

    st.sidebar.success("✅ Ready!")

# =========================
# KEYWORDS HELPER
# =========================

def get_keywords(question):
    if not question:
        return []
    for paper in DATASET:
        if question in paper["ground_truth"]:
            return paper["ground_truth"][question]
    return question.lower().split()[:3]

# =========================
# CHAT HISTORY
# =========================

for entry in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(entry["question"])
    for model_name, result in entry["results"].items():
        with st.chat_message("assistant"):
            st.markdown(f"### 🤖 {model_name}")
            st.markdown(f"**Answer:** {result['answer']}")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Answer Quality", round(result["aq"], 2))
            col2.metric("Faithfulness",   round(result["faith"], 2))
            col3.metric("Context Use",    round(result["cu"], 2))
            col4.metric("Latency (s)",    round(result["latency"], 2))
            with st.expander("📄 Retrieved Context"):
                st.write(result["context"])

# =========================
# SUGGESTION CHIPS
# Rendered BEFORE the text_area. On click we only touch `input_text`
# and `input_ver` (not the widget key), then rerun — safe.
# =========================

paper_data = next(p for p in DATASET if p["paper"] == selected_paper)

if st.session_state.rag_pipelines:
    st.caption("💡 Suggested questions — click to fill the input box:")
    cols = st.columns(len(paper_data["questions"]))
    for i, q in enumerate(paper_data["questions"]):
        if cols[i].button(q, key=f"chip_{i}", use_container_width=True):
            st.session_state.input_text = q
            st.session_state.input_ver += 1   # new key → new widget → value= is respected
            st.rerun()
else:
    st.info("👈 Load a paper from the sidebar to get started.")

# =========================
# INPUT BOX
# key changes whenever input_ver changes, so Streamlit treats it as a
# brand-new widget and honours the value= argument.
# =========================

widget_key = f"qa_input_{st.session_state.input_ver}"

user_text = st.text_area(
    "question_label",
    value=st.session_state.input_text,
    key=widget_key,
    height=90,
    placeholder="Type your question here, or pick a suggestion above ↑",
    label_visibility="collapsed"
)

send_clicked = st.button("➤ Send", type="primary")

# =========================
# RUN CHAT
# =========================

if send_clicked and user_text and user_text.strip():
    question_to_run = user_text.strip()

    if not st.session_state.rag_pipelines:
        st.warning("Please load a paper first using the sidebar.")
        st.stop()

    # Clear the box for next question (safe here — widget already rendered above)
    st.session_state.input_text = ""
    st.session_state.input_ver += 1

    with st.chat_message("user"):
        st.write(question_to_run)

    entry = {"question": question_to_run, "results": {}}

    for model_name, rag in st.session_state.rag_pipelines.items():
        with st.chat_message("assistant"):
            st.markdown(f"### 🤖 {model_name}")

            with st.spinner(f"{model_name} is thinking..."):
                start = time.time()
                output = rag.invoke(question_to_run)
                latency = time.time() - start

            if isinstance(output, dict):
                answer  = output.get("answer", "")
                context = output.get("context", "")
            else:
                answer  = output
                context = ""

            keywords = get_keywords(question_to_run)
            aq    = answer_quality(answer, keywords) if keywords else 0
            faith = faithfulness(answer, context)
            cu    = context_utilization(answer, context)

            st.markdown(f"**Answer:** {answer}")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Answer Quality", round(aq, 2))
            col2.metric("Faithfulness",   round(faith, 2))
            col3.metric("Context Use",    round(cu, 2))
            col4.metric("Latency (s)",    round(latency, 2))

            with st.expander("📄 Retrieved Context"):
                st.write(context)

            entry["results"][model_name] = {
                "answer": answer, "context": context,
                "aq": aq, "faith": faith, "cu": cu, "latency": latency
            }

    st.session_state.chat_history.append(entry)
    st.rerun()