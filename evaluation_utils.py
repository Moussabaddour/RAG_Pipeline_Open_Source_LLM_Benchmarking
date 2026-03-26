import pandas as pd

def build_results_table(results):

    df = pd.DataFrame(results)

    summary = df.groupby("model").agg({
        "answer_quality": "mean",
        "faithfulness": "mean",
        "context_utilization": "mean",
        "latency": "mean"
    }).reset_index()

    return df, summary