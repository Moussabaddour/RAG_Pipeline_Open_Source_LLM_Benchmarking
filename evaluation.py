import time
from sentence_transformers import SentenceTransformer, util

# -------------------------
# Load embedding model once
# -------------------------
_embedder = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)


# -------------------------
# 1. Answer Quality
# -------------------------
def answer_quality(answer: str, keywords: list[str]) -> float:
    if not answer.strip() or not keywords:
        return 0.0
    ground_truth = " ".join(keywords)
    answer_emb   = _embedder.encode(answer,       convert_to_tensor=True)
    gt_emb       = _embedder.encode(ground_truth, convert_to_tensor=True)
    return float(util.cos_sim(answer_emb, gt_emb))


# -------------------------
# 2. Faithfulness
# -------------------------
def faithfulness(answer: str, context: str) -> float:
    if not answer.strip() or not context.strip():
        return 0.0
    answer_emb  = _embedder.encode(answer,  convert_to_tensor=True)
    context_emb = _embedder.encode(context, convert_to_tensor=True)
    return float(util.cos_sim(answer_emb, context_emb))


# -------------------------
# 3. Context Utilization
# -------------------------
def context_utilization(answer: str, context: str) -> float:
    if not answer.strip() or not context.strip():
        return 0.0

    # Split on chunk boundaries (double or single newlines)
    sentences = [s.strip() for s in context.replace("\n\n", "\n").split("\n")]
    # Keep only meaningful sentences (≥ 5 words)
    sentences = [s for s in sentences if len(s.split()) >= 5]

    if not sentences:
        return 0.0

    answer_emb    = _embedder.encode(answer,    convert_to_tensor=True)
    sentence_embs = _embedder.encode(sentences, convert_to_tensor=True)

    scores = [float(util.cos_sim(answer_emb, s_emb)) for s_emb in sentence_embs]
    return sum(scores) / len(scores)


# -------------------------
# 4. Evaluation loop
# -------------------------
def evaluate_rag(rag_chain, dataset, model_name: str) -> list[dict]:

    results = []

    for paper_data in dataset:

        paper = paper_data["paper"]

        for question in paper_data["questions"]:

            start  = time.time()
            output = rag_chain.invoke(question)
            latency = time.time() - start

            if isinstance(output, dict):
                answer  = output.get("answer", "")
                context = output.get("context", "")
            else:
                answer  = output
                context = ""

            keywords = paper_data["ground_truth"][question]

            results.append({
                "model":               model_name,
                "paper":               paper,
                "question":            question,
                "answer":              answer,
                "latency":             latency,
                "answer_quality":      answer_quality(answer, keywords),
                "faithfulness":        faithfulness(answer, context),
                "context_utilization": context_utilization(answer, context),
            })

    return results