import os

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel

from sentence_transformers import SentenceTransformer


# -------------------------
# Load documents
# -------------------------
def load_documents(data_path="data/"):
    documents = []

    for file in os.listdir(data_path):
        path = os.path.join(data_path, file)

        if file.endswith(".pdf"):
            loader = PyPDFLoader(path)
        else:
            loader = TextLoader(path)

        documents.extend(loader.load())

    return documents


# -------------------------
# Split documents
# -------------------------
def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    return splitter.split_documents(documents)


# -------------------------
# LOCAL Embedding wrapper
# -------------------------
class LocalEmbedding:

    def __init__(self, model_path="all-MiniLM-L6-v2"):
        # local_files_only prevents any HuggingFace Hub network calls
        self.model = SentenceTransformer(model_path, local_files_only=True)

    def embed_documents(self, texts):
        return self.model.encode(texts, show_progress_bar=True)

    def embed_query(self, text):
        return self.model.encode([text])[0]

    def __call__(self, text):
        return self.embed_query(text)


# -------------------------
# Create vector DB
# -------------------------
def create_vectorstore(chunks):
    embedding_model = LocalEmbedding()
    vectorstore = FAISS.from_documents(chunks, embedding_model)
    return vectorstore


# -------------------------
# Format docs
# -------------------------
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# -------------------------
# Build RAG chain
# -------------------------
def build_rag_chain(llm, vectorstore):

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    prompt = PromptTemplate.from_template("""
[INST]
You are an expert AI research assistant.

Answer ONLY using the provided context.
- If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{question}
[/INST]
""")

    # Step 1: retrieve context and pass question through in parallel
    context_and_question = RunnableParallel({
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    })

    # Step 2: generate answer while preserving the context for evaluation
    def generate_with_context(inputs):
        context = inputs["context"]
        question = inputs["question"]
        answer = (prompt | llm).invoke({"context": context, "question": question})
        return {
            "answer": answer,
            "context": context
        }

    rag_chain = context_and_question | RunnableLambda(generate_with_context)

    return rag_chain