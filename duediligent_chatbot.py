#!/usr/bin/env python
# coding: utf-8

# In[9]:


import openai
import requests
import datetime
api_key = "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"  # Replace with your actual OpenAI API key

# API headers
headers = {'Authorization': f'Bearer {api_key}'}

# API endpoint
url = 'https://api.openai.com/v1/usage'

# Date for which to get usage data
date = datetime.date(2024, 1, 16)

# Parameters for API request
params = {'date': date.strftime('%Y-%m-%d')}

# Send API request and get response
response = requests.get(url, headers=headers, params=params)
usage_data = response.json()['data']

# Calculate total number of tokens used for each model
total_tokens_used_davinci = 0
total_tokens_used_ada = 0

for data in usage_data:
    model_name = data['model']
    n_generated_tokens_total = data['n_generated_tokens_total']
    n_context_tokens_total = data['n_context_tokens_total']
    total_tokens = n_generated_tokens_total + n_context_tokens_total
    if model_name == 'text-davinci-003':
        total_tokens_used_davinci += total_tokens
    elif model_name == 'text-embedding-ada-002':
        total_tokens_used_ada += total_tokens

# Estimate cost for each model based on token usage
davinci_cost_per_token = 0.002 / 1000
ada_cost_per_token = 0.0004 / 1000

total_cost_davinci = total_tokens_used_davinci * davinci_cost_per_token
total_cost_ada = total_tokens_used_ada * ada_cost_per_token

# Print estimated costs
print(f"Total number of tokens used by text-davinci-003 on {date}: {total_tokens_used_davinci}")
print(f"Estimated cost for text-davinci-003 on {date}: ${total_cost_davinci:.2f}")

print(f"\nTotal number of tokens used by text-embedding-ada-002 on {date}: {total_tokens_used_ada}")
print(f"Estimated cost for text-embedding-ada-002 on {date}: ${total_cost_ada:.2f}")


# In[6]:



import streamlit as st
from llama_index import VectorStoreIndex, ServiceContext, Document
from llama_index.llms import OpenAI
import openai
from llama_index import SimpleDirectoryReader


# In[10]:


openai.api_key = "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"  # Replace with your actual OpenAI API key
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
            


# In[ ]:




