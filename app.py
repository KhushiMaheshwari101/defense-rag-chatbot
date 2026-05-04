import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings # Updated import
from langchain_community.retrievers import BM25Retriever
from langchain_groq import ChatGroq
from sentence_transformers import CrossEncoder
import tempfile
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Defense RAG Assistant", page_icon="🛡️")
st.title("🛡️ Defense RAG Assistant")

# --- API KEY SECURITY ---
# Local pe chalaogi toh .env se lega, Streamlit pe secrets se lega
if "GROQ_API_KEY" in st.secrets:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
else:
    # Testing ke liye agar local pe chala rahi ho
    GROQ_API_KEY = "PASTE_YOUR_KEY_HERE_ONLY_FOR_LOCAL_TESTING"

# --- CACHING MODELS (Taaki app baar baar load na ho) ---
@st.cache_resource
def load_models():
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return reranker, embeddings

reranker, embeddings = load_models()

def get_section_hint(query):
    query_lower = query.lower()
    hints = {
        "perspectives of war": "Section 3.2-3.5 - Strategic Perspective, Operational Perspective, Tactical Perspective",
        "principles of war": "Section 2.24-2.35 - Selection of Aim, Morale, Offensive Action, Surprise, Concentration, Security, Economy of Effort, Flexibility, Cooperation, Administration, Intelligence",
        "modes of combat power": "Section 3.11 - Destruction, Attacking Enemy Will, Pre-emption, Dislocation, Disruption",
        "types of war": "Section 5 - Conventional War, OOTW, Sub-conventional, Nuclear, Chemical, Biological",
        "information warfare": "Section 2.14-2.23 - C2W, IBW, EW, Psychological, Cyber, Economic, NCW",
        "combat power": "Section 3.17-3.20 - Conceptual, Moral, Physical components",
        "roles of indian army": "Section 1.13 - Primary Role, Secondary Role",
        "tasks of indian army": "Section 1.15 - Tasks listed",
        "rma": "Section 3.37-3.45 - Revolution in Military Affairs",
        "surprise": "Section 3.23-3.24 - Surprise in war",
        "deception": "Section 3.25-3.28 - Deception operations",
        "insurgency": "Section 2.9 - Insurgency definition",
        "proxy war": "Section 2.8 - Proxy war definition",
        "low intensity conflict": "Section 2.7 - LIC definition",
    }
    for key, value in hints.items():
        if key in query_lower:
            return value
    return ""

# --- MAIN APP ---
uploaded_file = st.file_uploader("Upload Defense Document (PDF)", type="pdf")

if uploaded_file:
    # PDF Processing (Only once using Session State)
    if "vectorstore" not in st.session_state:
        with st.spinner("Processing Document..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
                f.write(uploaded_file.read())
                temp_path = f.name

            loader = PyPDFLoader(temp_path)
            documents = loader.load()

            splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=400)
            chunks = splitter.split_documents(documents)

            st.session_state.vectorstore = Chroma.from_documents(chunks, embeddings)
            st.session_state.bm25 = BM25Retriever.from_documents(chunks)
            st.session_state.bm25.k = 10
            st.session_state.num_chunks = len(chunks)
            os.remove(temp_path) # Clean up temp file

    st.success(f"Document ready with {st.session_state.num_chunks} chunks!")

    # Chat UI
    llm = ChatGroq(api_key=GROQ_API_KEY, model_name="llama-3.1-8b-instant")
    query = st.text_input("Ask anything from document:")

    if query:
        with st.spinner("Analyzing Defense Doctrine..."):
            # Hybrid Search
            semantic_docs = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 10}).invoke(query)
            bm25_docs = st.session_state.bm25.invoke(query)

            # Deduplicate
            all_docs = list({doc.page_content: doc for doc in semantic_docs + bm25_docs}.values())

            # Re-ranking
            pairs = [[query, doc.page_content] for doc in all_docs]
            scores = reranker.predict(pairs)
            ranked = sorted(zip(scores, all_docs), key=lambda x: x[0], reverse=True)
            
            top_docs = [doc for _, doc in ranked[:6]]
            context = "\n\n".join([doc.page_content for doc in top_docs])
            section_hint = get_section_hint(query)

            prompt = f"""You are a military defense expert assistant analyzing the Indian Army Doctrine 2004 document.
            STRICT RULES:
            1. Answer ONLY from the context provided below.
            2. List ALL points completely — never skip any point.
            3. Never confuse similar topics.
            4. Detailed explanation for each point.

            {f"SECTION HINT: Look specifically for {section_hint}" if section_hint else ""}

            Context: {context}
            Question: {query}
            Detailed Answer:"""

            response = llm.invoke(prompt)
            st.subheader("🎯 Answer:")
            st.write(response.content)