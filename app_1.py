################################################################################
# Author : Rohan Mohapatra
# Email : rohanmohapatra100@gmail.com
################################################################################

from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Literal,Annotated
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph.message import add_messages
from langchain_huggingface import HuggingFaceEmbeddings, ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
from PyPDF2 import PdfReader
import os
import streamlit as st



# ENTER YOUR HUGGING FACE API KEY HERE

########## Prompt Templates ###########

template1 = PromptTemplate(template='''you are an helpful research assistant and an supergenius in physics having an conversation.
                           the conversation history is: {history}
                          give answer to the current query: {query} .
                          form the base of your answer from the given context: {context}
                          if the context is insufficient, answer the query after stating that the knowledge base is insufficient''',
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

template_chat = PromptTemplate(template='''you are a genius physics reviewer and you are reviewing a content based on some query.
                               properly organize the content and complete all the mathematical steps and explainations with extreme clarity and reasoning.
                               if the content is small and the query indicates normal chatting continue the chat in a friendly manner and ask how you can help based on the input you recieved.
                               the content is: {content}
                               the query is: {query}''',
                          input_variables= [ "content", "query"])


def format_history(messages):
    formatted = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            formatted.append(f"Human: {msg.content}")
        elif isinstance(msg, AIMessage):
            formatted.append(f"Assistant: {msg.content}")
    return "\n".join(formatted)

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

########### Schema ###########

class chat_state(TypedDict): 

    messages: Annotated[list[BaseMessage], add_messages]
    messages_chat: Annotated[list[BaseMessage], add_messages]
    retrived: list
    intent: Literal["chat", "numeric", "summarize"]
    summary: str
    solution: str
    pdf_text: str



if "graph_state" not in st.session_state:
    st.session_state.graph_state = {
        "messages": [],
        "messages_chat": [],
        "retrived": [],
        "intent": "chat",
        "summary": "",
        "solution": "",
        "pdf_text": ""
    }

st.sidebar.title("🧭 Choose Mode")
mode = st.sidebar.radio(
    "Select what you want to do:",
    ["Select mode","Chat about physics", "Solve numericals", "Summarize research paper"]
)
MODE_TO_INTENT = {
    "Chat about physics": "chat",
    "Solve numericals": "numeric",
    "Summarize research paper": "summarize"
}

intent = MODE_TO_INTENT.get(mode, None)

if intent is not None:
    st.session_state.graph_state["intent"] = intent


########### Describing Models #############
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"device": "cpu"}
)

vectordb = FAISS.load_local(
    folder_path="./faiss_db",
    embeddings=embedding_model,
    allow_dangerous_deserialization=True
)

llm1 = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1",
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

llm_chat = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

model1 = ChatHuggingFace(llm=llm1)
model2 = ChatHuggingFace(llm=llm2)
model3 = ChatHuggingFace(llm=llm3)
model_chat = ChatHuggingFace(llm=llm_chat)


retriever2 = vectordb.as_retriever(search_type = "mmr", search_kwargs={"k": 5})

retriever_tool = create_retriever_tool( retriever=retriever2, 
                                       name="physics_knowledge_base", 
                                       description=( '''Retrieve relevant physics and mathematics theory, 
                                                    definitions and derivations from textbooks.''' ) ) 
model1_tool = model1.bind_tools([retriever_tool])



########## Describing Nodes ##########




def base_node(state: chat_state):
    print("reached base node")
    query = state["messages"][-1].content
    retrived_docs =retriever2.invoke(query)
    # print(query)

    return {"retrived" : retrived_docs}

def retriver(state: chat_state):
    print("reached retriver node")
    # prompt = template1.invoke({"history" : format_history(state["messages_chat"]), "query" : state["messages_chat"][-1].content, "context"  : full_context(state["retrived"])})
    # answer = model1.invoke(prompt)
    answer = model1_tool.invoke(state["messages"])
    return {"messages": state["messages"] + [answer]}

def numeric_solver(state: chat_state):
    print("reached solver node")
    prompt = template2.invoke({"history" : format_history(state["messages_chat"]), "query" : state["messages_chat"][-1].content, "context"  : full_context(state["retrived"])})
    answer = model2.invoke(prompt)
    return {"messages": state["messages"] + [answer]}

def summarizer(state: chat_state):
    print("reached summerizer node")
    prompt = template3.invoke({ "context"  : state["pdf_text"]})
    answer = model3.invoke(prompt)
    return {"messages": state["messages"] + [answer]}

def chat_node(state: chat_state):
    prompt = template_chat.invoke({"content" : state["messages"][-1].content, "query": state["messages_chat"][-1].content})
    answer = model_chat.invoke(prompt)
    
    return {"messages_chat": state["messages_chat"] + [answer]}


Route = Literal["Retriever", "Numeric_solver", "Summarizer", "end"]

def check_condition(state: chat_state) -> Route:
    if state["intent"] == "chat":
        return "Retriever"
    elif state["intent"] == "numeric":
        return "Numeric_solver"
    else:
        return "Summarizer"


############ Describe the graph ###########

graph = StateGraph(chat_state)

graph.add_node("Base_Node", base_node)
graph.add_node("Chat_Node", chat_node)
graph.add_node("Retriever", retriver)
graph.add_node("Numeric_solver", numeric_solver)
graph.add_node("Summarizer", summarizer)

graph.add_edge(START, "Base_Node")
graph.add_conditional_edges("Base_Node", check_condition,{
        "Retriever": "Retriever",
        "Numeric_solver": "Numeric_solver",
        "Summarizer": "Summarizer",
    })

graph.add_edge("Retriever", "Chat_Node")
graph.add_edge("Numeric_solver", "Chat_Node")
graph.add_edge("Summarizer", "Chat_Node")
graph.add_edge("Chat_Node", END)

workflow = graph.compile()



########### Chat process ############

st.title("👨‍🚀 Physics RAG Chatbot")

for msg in st.session_state.graph_state["messages"]:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(msg.content)
query = st.chat_input("Ask your question or type 'exit' to stop...")

if st.session_state.graph_state["intent"] == "summarize":
        uploaded_file = st.file_uploader("Upload a research paper (PDF)", type=["pdf"])
        if uploaded_file:
            st.markdown("File uploaded ")
            text = extract_text_from_pdf(uploaded_file)
            st.session_state.graph_state["intent"] = text
if query:
    if query.strip().lower() == "exit":
        st.success("Goodbye! 👋")
        st.stop()

    with st.chat_message("user"):
            st.markdown(query)


    st.session_state.graph_state["messages"].append(HumanMessage(content=query))
    st.session_state.graph_state["messages_chat"].append(HumanMessage(content=query))
    
    st.session_state.graph_state = workflow.invoke(st.session_state.graph_state)

    # Print latest model response
    if st.session_state.graph_state.get("messages_chat"):
        with st.chat_message("assistant"):
            st.markdown(st.session_state.graph_state["messages_chat"][-1].content)




