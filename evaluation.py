import time
 
 
# -------------------------
# 1. Answer Quality
# -------------------------
def answer_quality(answer, keywords):
    answer = answer.lower()
    hits = sum(1 for kw in keywords if kw.lower() in answer)
    return hits / len(keywords)
 
 
# -------------------------
# 2. Faithfulness
# -------------------------
def faithfulness(answer, context):
    answer_words = set(answer.lower().split())
    context_words = set(context.lower().split())
 
    overlap = answer_words.intersection(context_words)
    return len(overlap) / max(len(answer_words), 1)
 
 
# -------------------------
# 3. Context Utilization
# -------------------------
def context_utilization(answer, context):
    """
    Measures how much of the retrieved context is reflected in the answer.
    For each meaningful context sentence, computes word-overlap with the answer.
    Returns the average overlap score across all non-trivial sentences.
    """
    answer_words = set(answer.lower().split())
 
    # Split on double newlines (chunk boundaries) then further on single newlines
    sentences = [s.strip() for s in context.replace("\n\n", "\n").split("\n")]
 
    # Keep only sentences with at least 5 words (filter out headers/fragments)
    sentences = [s for s in sentences if len(s.split()) >= 5]
 
    if not sentences:
        return 0.0
 
    scores = []
    for sent in sentences:
        sent_words = set(sent.lower().split())
        overlap = len(answer_words & sent_words)
        scores.append(overlap / max(len(sent_words), 1))
 
    return sum(scores) / len(scores)
 
 
# -------------------------
# 4. Evaluation loop
# -------------------------
def evaluate_rag(rag_chain, dataset, model_name):
 
    results = []
 
    for paper_data in dataset:
 
        paper = paper_data["paper"]
 
        for question in paper_data["questions"]:
 
            start = time.time()
 
            output = rag_chain.invoke(question)
 
            latency = time.time() - start
 
            # answer = output["answer"]
            # context = output["context"]
            if isinstance(output, dict):
                answer = output.get("answer", "")
                context = output.get("context", "")
            else:
                answer = output
                context = ""
                
            keywords = paper_data["ground_truth"][question]
 
            results.append({
                "model": model_name,
                "paper": paper,
                "question": question,
                "answer": answer,
                "latency": latency,
                "answer_quality": answer_quality(answer, keywords),
                "faithfulness": faithfulness(answer, context),
                "context_utilization": context_utilization(answer, context)
            })
 
    return results