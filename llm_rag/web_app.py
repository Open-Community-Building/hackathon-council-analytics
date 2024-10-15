import streamlit as st
from query import RAG_LLM


st.title("Council Agenda Analytics Chatbot")
st.header("Ask anything!")
user_input = st.text_input("Enter your question:")

rag_llm = RAG_LLM()

if st.button("Get Response"):
    if user_input:
        response = rag_llm.query_rag_llm(user_input)

        # Display the response
        st.markdown("### Response from the Chatbot:")
        st.success(response)

        # Optionally, you can show chat history
        st.markdown("### Chat History:")
        st.markdown(f"**You**: {user_input}")
        st.markdown(f"**Chatbot**: {response}")
    else:
        st.error("Please enter a question!")
