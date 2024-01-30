#!/usr/bin/env python
# coding: utf-8

import streamlit as st
from llama_index import VectorStoreIndex, ServiceContext, Document
from llama_index.llms import OpenAI
import openai
from llama_index import SimpleDirectoryReader


openai.api_key = ""  # Replace with your actual OpenAI API key
# Initialize message history
st.header("Chat with DueDiligent AI 💬 📊")

if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "Ask me a question about DueDiligent's AI for due diligence process!"}]

@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Loading and indexing DueDiligent AI data – please wait."):
        reader = SimpleDirectoryReader(input_dir="C:\\Users\\shaur\\OneDrive\\Desktop\\Dueligent\\", recursive=True)  # Adjust the data directory accordingly
        docs = reader.load_data()
        service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo", temperature=0.5, system_prompt="You are an expert in due diligence processes, specializing in legal, financial, technical, and market analysis. Your role is to provide insightful and accurate information related to due diligence inquiries. Please answer questions based on your expertise in these areas.  do not hallucinate features."))
        index = VectorStoreIndex.from_documents(docs, service_context=service_context)
        return index

duediligent_index = load_data()

chat_engine = duediligent_index.as_chat_engine(chat_mode="condense_question", verbose=True)

# Prompt for user input and display message history
if prompt := st.chat_input("Your question"):
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Pass query to chat engine and display response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Use LlamaIndex's chat engine to generate a response
            response = chat_engine.chat(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message)
            



