#!/usr/bin/env python
# coding: utf-8
import streamlit as st
import openai
import plotly.express as px
from llama_index import VectorStoreIndex, ServiceContext, Document
from llama_index.llms import OpenAI
from llama_index import SimpleDirectoryReader

# Set your OpenAI API key
openai.api_key = ""

# Initialize message history as a global list
message_history = []

# Initialize Streamlit app
st.header("Chat with DueDiligent AI 💬 📊")

# Define a function to generate a plot based on knowledge from OpenAI
def generate_plot_from_openai_knowledge(response):
    # Extract relevant information from OpenAI response
    # Modify this part based on your actual response structure
    knowledge = response.choices[0].text

    # Generate a plot based on the extracted knowledge
    # Here's an example using Plotly Express:
    df = px.data.gapminder()
    fig = px.scatter(df, x="year", y="gdpPercap", color="continent", title=f"Plot based on OpenAI Knowledge:\n{knowledge}")
    return fig

# Initialize the chat engine as in your original code
@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Loading and indexing DueDiligent AI data – please wait."):
        reader = SimpleDirectoryReader(input_dir="C:\\Users\\shaur\\OneDrive\\Desktop\\Dueligent\\", recursive=True)
        docs = reader.load_data()
        service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo", temperature=0.5, system_prompt="You are an expert in due diligence processes, specializing in legal, financial, technical, and market analysis. Your role is to provide insightful and accurate information related to due diligence inquiries. Please answer questions based on your expertise in these areas.  do not hallucinate features"))
        index = VectorStoreIndex.from_documents(docs, service_context=service_context)
        return index

duediligent_index = load_data()

chat_engine = duediligent_index.as_chat_engine(chat_mode="condense_question", verbose=True)

# Prompt for user input and display message history
user_input = st.text_input("You: ")
if st.button("Ask"):
    if user_input:
        # User message
        message_history.append({"role": "user", "content": user_input})

        # Check if the user is requesting a plot
        if "plot" in user_input.lower():
            # Generate a plot based on OpenAI knowledge
            with st.spinner("Thinking..."):
                try:
                    response = openai.Completion.create(
                        engine="text-davinci-003",
                        prompt=user_input,
                        max_tokens=150
                    )

                    response_text = response.choices[0].text.strip()
                    message_history.append({"role": "assistant", "content": response_text})

                    # Generate a plot based on OpenAI knowledge
                    plot = generate_plot_from_openai_knowledge(response)
                    st.plotly_chart(plot)

                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")
        else:
            # Append the user's message to the message history
            message_history.append({"role": "user", "content": user_input})

# Check if the last role is not assistant to trigger a response
if message_history and message_history[-1]["role"] != "assistant":
    with st.spinner("Thinking..."):
        try:
            # Use OpenAI GPT-3.5 turbo to generate a response
            response = chat_engine.chat(user_input)  # Use chat_engine instead of openai.Completion
            response_text = response.response
            st.text(f"Bot: {response_text}")
            message_history.append({"role": "assistant", "content": response_text})
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")

# Display message history
for message in message_history:
    if message["role"] == "user":
        st.text(f"You: {message['content']}")
    elif message["role"] == "assistant":
        st.text(f"DueDiligent AI: {message['content']}")
