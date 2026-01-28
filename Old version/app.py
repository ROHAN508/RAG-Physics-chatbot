import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings, ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from PyPDF2 import PdfReader
import os

# SET YOU ENVIRONMENT API KEY HERE


if "messages" not in st.session_state:
    st.session_state.messages = []
if "mode" not in st.session_state:
    st.session_state.mode = None

st.sidebar.title("🧭 Choose Mode")
mode = st.sidebar.radio(
    "Select what you want to do:",
    ["Select mode","Chat about physics", "Solve numericals", "Summarize research paper"]
)
st.session_state.mode = mode

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

vectordb = FAISS.load_local(
    folder_path="./faiss_db",
    embeddings=embedding_model,
    allow_dangerous_deserialization=True
)

llm1 = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation"
)

llm2 = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1",
    task="text-generation"
)

llm3 = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1",
    task="text-generation"
)

model1 = ChatHuggingFace(llm=llm1)
model2 = ChatHuggingFace(llm=llm2)
model3 = ChatHuggingFace(llm=llm3)

retriever1 = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 5})
retriever2 = vectordb.as_retriever(search_type = "mmr", search_kwargs={"k": 5})

def full_context(docs1, docs2 = []):
    all_docs = docs1 +docs2
    context_text = "\n\n".join(doc.page_content for doc in all_docs)
    return context_text

def extract_text_from_pdf(pdf_file):
            reader = PdfReader(pdf_file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text


template1 = PromptTemplate(template='''you are an helpful research assistant and an supergenius in physics having an conversation.
                           the conversation history is: {history}
                          give answer to the current query: {query} .
                          form the base of your answer from the given context: {context}
                          if the context is insufficient answer say that the knowledge base is insufficient''',
                          input_variables= ["history", "query", "context"])

template2 = PromptTemplate(template='''you are an helpful research assistant and an supergenius in physics having an conversation.
                           the conversation history is: {history}
                          Solve the current numerical: {query} .
                          given theoretical context is: {context}
                          if the context is misleading ignore it''',
                          input_variables= ["history", "query", "context"])

template3 = PromptTemplate(template='''you are a genius university professor in physics and a student asks you to summarize a research paper.
                           the research paper context is: {context}
                          summarize it with extreme clarity and with proper mathematical reasoning''',
                          input_variables= [ "context"])

st.title("👨‍🚀 Physics RAG Chatbot")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if mode == "Select mode":
        st.markdown("Please select a mode...")
elif mode == "Summarize research paper":
        uploaded_file = st.file_uploader("Upload a research paper (PDF)", type=["pdf"])
        if uploaded_file:
            st.markdown("File uploaded ")
            text = extract_text_from_pdf(uploaded_file)
            prompt = template3.invoke({ "context"  : text})
            answer = model3.invoke(prompt)
            with st.chat_message("assistant"):
                st.markdown(answer.content)
else:
    query = st.chat_input("Ask your question or type 'exit' to stop...")

    if query:
        if query.lower() == "exit":
            st.success("Goodbye! 👋")
            st.stop()

        with st.chat_message("user"):
            st.markdown(query)

        st.session_state.messages.append({"role": "user", "content": query})
        history = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in st.session_state.messages[-6:]])

        if mode == "Chat about physics":
            retrived_docs_1 =retriever1.invoke(query)
            retrived_docs_2 =retriever2.invoke(query)
            prompt = template1.invoke({"history" : history, "query" : query, "context"  : full_context(retrived_docs_1, retrived_docs_2)})
            answer = model1.invoke(prompt)

        if mode == "Solve numericals":
            retrived_docs =retriever2.invoke(query)
            prompt = template2.invoke({"history" : history, "query" : query, "context"  : full_context(retrived_docs)})
            answer = model2.invoke(prompt)


        with st.chat_message("assistant"):
            st.markdown(answer.content)