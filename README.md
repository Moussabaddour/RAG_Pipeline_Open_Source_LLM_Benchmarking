# RAG Pipeline & Open-Source LLM Benchmarking

**Author**: Moussa BADDOUR

## Overview

This project implements a Retrieval-Augmented Generation (RAG) pipeline for evaluating and benchmarking open-source Large Language Models (LLMs).

It includes:

- Document loading and preprocessing
- Vector database creation
- RAG-based question answering
- Evaluation metrics (answer quality, faithfulness, context utilization)
- Interactive Streamlit dashboard

## Setup Instructions

1. Clone the repository

        git clone <repository-url>
        cd <repository-folder>

2. Model Downloads

    
    Download the models from the following links:

    🔹 LLM Models: Download the language models used in the RAG pipeline:

            
                https://filesender.renater.fr/?s=download&token=e1702004-d474-4930-bd60-24cd1083c83f


    🔹 Embedding Model: Download the embedding model used to convert text into vectors for the FAISS vector store

                https://filesender.renater.fr/?s=download&token=e1702004-d474-4930-bd60-24cd1083c83f
    
    After downloading, place all models in the models/

2. Create a virtual environment (Python 3.10 recommended)

    The project was tested on **Linux (Ubuntu) and WSL**, but should work on any system with Python 3.10.
    
    Option A — Using venv

        sudo apt install python3.10-venv
        python3 -m venv <environment_name>
        source <environment_name>/bin/activate
    
    Option B — Using Conda (Recomended)

        conda create --name <your-env> python=3.10
        conda activate <your-env>

3. Install dependencies

        pip install -r requirements.txt

## Usage
    
- Run the main pipeline
    
    
        python main.py (This process may take approximately 25 minutes using CPU, as it evaluates all questions across 5 models.) 
        
        
        Optional: GPU Acceleration

        
        You can enable GPU acceleration by modifying the following parameter in llm_loader.py:

                n_gpu_layers = 0
                
                        -  0 → CPU only (default, stable)
                
                        -  1 → full GPU usage (all layers on GPU)
                
                        -  N → partial GPU (e.g., 20 layers on GPU, rest on CPU)

        ⚠️ Note:

        GPU usage depends on your hardware and CUDA setup.
        For stability in the Streamlit app (app.py), it is recommended to reset n_gpu_layers = 0 to avoid context conflicts during model reload.

OR

- Launch the web interface


        streamlit run app.py
        
The application will be available at http://localhost:8501/


Using the Web Interface

1- Select a paper
        
        Choose the document to be used as input for the RAG system.

2- Select model(s)
        
        Choose one or multiple LLMs for evaluation.

3- Load the system
        
        Click "Load Selected Paper" to initialize the RAG pipeline and models.

4- Ask a question
        
        Select a predefined question (grounded evaluation)
        OR
        write your own question in the text field (may be less grounded)

5-  View results
        
        1. Generated answer

        2. Evaluation metrics

                - Answer Quality

                - Faithfulness

                - Context Utilization

                - Latency.