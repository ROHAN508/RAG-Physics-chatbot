# 🧠 Physics RAG Chatbot using LangChain, LangGraph & OpenSource LLM

### Author:
Rohan Mohapatra,  
Integrated M.Sc. Physics & Astronomy, NIT Rourkela    
Interested in Machine Learning, Quantum Computing and physics

This project is a physics-focused Retrieval-Augmented Generation (RAG) chatbot built using **LangGraph**, **LangChain**, **FAISS**, and **Open Source LLMs**.  
The system supports conversational physics queries, numerical problem solving, and research paper summarization through a structured graph-based workflow.

---

## 🚀 Features

- 🔍 **Physics-aware RAG** using FAISS vector database  
- 🧠 **Graph-based reasoning pipeline** using LangGraph  
- 💬 **Conversational physics chat**
- ✏️ **Step-by-step numerical problem solving**
- 📄 **Research paper summarization (PDF input)**
- 🧩 **Intent-based routing** (chat / numericals / summarization)
- 🖥️ **Interactive Streamlit UI**
- 🔒 **Open Source LLMs**

---

## Tech Stacks used:

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?logo=chainlink&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-0A0A0A?logo=graphql&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-0467DF?logo=meta&logoColor=white)
![SentenceTransformers](https://img.shields.io/badge/SentenceTransformers-FF6F00?logo=huggingface&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![PyPDF2](https://img.shields.io/badge/PyPDF2-003B57?logo=adobeacrobatreader&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?logo=jupyter&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?logo=huggingface&logoColor=black)
![DeepSeek](https://img.shields.io/badge/DeepSeek-000000?logo=deepseek&logoColor=white)



## System Description
The newer version of the chatbot "app_1.py" is upgraded to Langgraph based implementation from the older Langchain based version.
This upgrade introduces a graph-driven control flow that explicitly models the reasoning pipeline as a state machine rather than a linear chain.  
Each user query is processed through a well-defined sequence of nodes, enabling better modularity, interpretability, and control over how information flows through the system.

The memory feature was implemented by storing the conversation data in the session_state of Streamlit UI, which was recursively used to give contexts using different prompts to the LLMs.

Altough every node which requires and LLM call uses the Same model "Deepseek-R1", it can be modified throuch the code with minimal changes to accommodate task specific fine tuned LLMs. The choice of using this particular model was made as it was one of the best models availabe free of cost.

At a high level, the system operates as follows:

1. **User Interaction Layer (Streamlit UI)**  
   The user selects a task mode (physics chat, numerical problem solving, or research paper summarization) and submits a query or document. This choice determines the intent stored in the global graph state.

2. **Intent-Aware Routing (LangGraph)**  
   Based on the selected intent, the LangGraph workflow conditionally routes the input to specialized nodes responsible for retrieval, numerical reasoning, or summarization.

3. **Retrieval-Augmented Generation (RAG)**  
   A large corpus of physics and mathematicsbased books avilable on the internet was created, Which was then chunked and stored in form of FAISS vectorstore.
   Retrival was performed as a tool calling feature based on the requirement of the Language model.

5. **Task-Specific Reasoning Nodes**  
   Separate nodes handle conversational explanations, step-by-step numerical solutions, and research paper summarization. Each node uses task-appropriate prompting strategyvto ensure clarity and domain relevance.

6. **Response Consolidation and Presentation**  
   The generated response is passed to a final chat node, which formats and presents the output consistently in the user interface.


## 🏗️ System Architecture

- User input → Base Node  
- Conditional routing:
  - Physics chat → Retriever
  - Numerical problems → Solver
  - Research paper → Summarizer  
- Final response → Chat node → UI

📌 **Graph structure:**

![LangGraph Structure](Screenshots/graph_architecture.png)


## 🖥️ User Interface

The frontend is built using **Streamlit**, allowing users to:

- Select the task mode from the sidebar
- Chat naturally about physics concepts
- Solve numericals step-by-step
- Upload PDFs for summarization

📌 **Example UI views:**

### Physics Chat
![Physics Chat UI](Screenshots/Chatting.png)

### Numerical Solver
![Numerical Solver UI](Screenshots/solving_numericals.png)

### Research Paper Summarization
![Summarization UI](Screenshots/Research_paper_summary.png)

---
## Future Improvements

Multi-document summarization  
Multimodal capabilities Including Image besed query processing and OCR  
More tool integratin to induce more agentic behaviour  
Evaluation on physics benchmark datasets  

## 📂 Repository Structure

```text
.
├── app_1.py                    # Main Streamlit + LangGraph application
├── RAG_data_ingestion.ipynb    # Data ingestion and FAISS index creation
├── retrival_logic.ipynb        # Retrieval and RAG experimentation
├── README.md
├── Old version/                # Archived experimental code




