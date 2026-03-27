RAG Pipeline & Open-Source LLM Benchmarking

Author: Moussa BADDOUR

Overview

This project implements a Retrieval-Augmented Generation (RAG) pipeline for evaluating and benchmarking open-source Large Language Models (LLMs).

It includes:

- Document loading and preprocessing
- Vector database creation
- RAG-based question answering
- Evaluation metrics (answer quality, faithfulness, context utilization)
- Interactive Streamlit dashboard

Setup Instructions

1. Clone the repository

        git clone <repository-url>
        cd <repository-folder>

2. Model Downloads

    Download the models from the following links:

    🔹 LLM Models
            Download the language models used in the RAG pipeline:

            
                https://filesender.renater.fr/?s=download&token=e1702004-d474-4930-bd60-24cd1083c83f


    🔹 Embedding Model
            Download the embedding model used to convert text into vectors for the FAISS vector store

                https://filesender.renater.fr/?s=download&token=e1702004-d474-4930-bd60-24cd1083c83f
    
    After downloading, place all models in the models/

2. Create a virtual environment (Python 3.10 recommended)
    
    Option A — Using venv

        sudo apt install python3.10-venv
        python3 -m venv <environment_name>
        source <environment_name>/bin/activate
    
    Option B — Using Conda

        conda create --name <your-env> python=3.10
        conda activate <your-env>

3. Install dependencies

        pip install -r requirements.txt

Usage
    
    - Run the main pipeline


        python main.py

    OR

    - Launch the web interface


        streamlit run app.py
        
        The application will be available at http://localhost:8501/


