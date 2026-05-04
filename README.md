 ---

#  Defense RAG Assistant
### *Intelligence-Driven Retrieval for Military Doctrine Analysis*

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![LangChain](https://img.shields.io/badge/Framework-LangChain-green?style=for-the-badge)
![Groq](https://img.shields.io/badge/LLM-Llama--3.1--8B-orange?style=for-the-badge)
![Status](https://img.shields.io/badge/Deployment-Live-brightgreen?style=for-the-badge)

##  Overview
Standard Large Language Models (LLMs) often struggle with specific military terminologies and can hallucinate when analyzing dense, sensitive doctrines. The **Defense RAG Assistant** is a specialized AI tool engineered to parse the *Indian Army Doctrine (2004)* with high precision. It utilizes a **Hybrid Search + Neural Re-ranking** architecture to ensure every response is strictly grounded in official documentation.

---

##  Key Technical Features
What makes this system "Defense-Grade"?

*   **Hybrid Retrieval Pipeline:** Combines **BM25 (Keyword-based)** search with **ChromaDB (Semantic-based)** search. This ensures that the system understands both the specific military terms and the underlying context.
*   **Neural Re-ranking:** Uses a `Cross-Encoder` model (`ms-marco-MiniLM-L-6-v2`) to evaluate query-context pairs, ensuring only the top 6 most relevant document chunks reach the LLM.
*   **Tactical Logic Integration:** Programmed with "Section Hints" to distinguish between easily confused concepts, such as *Strategic* vs. *Tactical* perspectives or different *Modes of Combat Power*.

---

##  Technical Workflow

1.  **Ingestion:** PDF documents are processed and split into optimized chunks (Size: 2000, Overlap: 400).
2.  **Embedding:** Text is converted into high-dimensional vectors using HuggingFace's `all-MiniLM-L6-v2` model.
3.  **Storage:** Vectors are stored in a high-performance `ChromaDB` vector store.
4.  **Querying:** A dual-stream retrieval process fetches the best matches from both semantic and keyword indices.
5.  **Generation:** The `Llama-3.1-8B` model (via Groq) synthesizes the final answer based strictly on the retrieved context.

---

##  Quick Start

###  Live Demo
[**Access the Assistant Here**](https://defense-rag-chatbot-qqbrv48gjeqxbdnojpdchp.streamlit.app/) 


###  Local Setup
1. **Clone & Install:**
   ```bash
   git clone [https://github.com/KhushiMaheshwari101/defense-rag-chatbot.git](https://github.com/KhushiMaheshwari101/defense-rag-chatbot.git)
   cd defense-rag-chatbot


---


###  Local Setup
1. **Clone & Install:**
   ```bash
   git clone https://github.com/KhushiMaheshwari101/defense-rag-chatbot.git
   cd defense-rag-chatbot
---

2. **Install dependencies**
   ```bash   
   pip install -r requirements.txt
---

3. **Add environmental configurations**
   Add your GROQ_API_KEY to your environment variables or Streamlit Secrets.
   
----
4. **Run the Application:**
   ```bash
   streamlit run app.py
   
---   


## Use Cases
* **Military Education:** Rapid reference for officers studying army doctrines and tactical manuals.
* **Strategic Research:** Deep-dive analysis into topics like Proxy War, RMA (Revolution in Military Affairs), and Information Warfare.
* **Academic Support:** Assisting defense students in summarizing and extracting critical points from complex military documents.

---
## Developer
**Khushi Maheshwari**
B.Tech Engineering Student | Guru Gobind Singh Indraprastha University
Specialization: Robotics and AI Research

---

## License
This project is for educational and research purposes. All doctrine content is understand.

