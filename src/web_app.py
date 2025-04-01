#streamlit run src/web_app.py ~/.config/hca/ --server.port=8686  --server.address=0.0.0.0
import streamlit as st
import sys
import os
from ragllm import RagLlm
import tomllib

# Define Defaults
DOCKER_CONFIGDIR = "/config"

if len(sys.argv) >= 2:
    configdir = sys.argv[1]
else:
    configdir = DOCKER_CONFIGDIR

config = None
st_title = "Council Agenda Analytics Chatbot"
st_header = "Ask anything!"
st_user_input = "Enter your question:"
st_get_response = "Get Response"
st_chbt_response = "### Response from the Chatbot:"
st_chbt_history = "### Chat History:"
st_history_input  = "**You**"
st_history_output = "**Chatbot**"
st_error_text = "Please enter a question!"


@st.cache_resource
def load_rag_llm(config: dict, secrets: dict):
    return RagLlm(config=config,secrets=secrets)

def read_config(configfile: str) -> dict:
    with open(configfile, "rb") as f:
        config = tomllib.load(f)
    return config

config = read_config(os.path.join(configdir, 'config.toml'))
secrets = read_config(os.path.join(configdir, 'secrets.toml'))



if config and config.get('streamlit'):
    # We have a configuration and will override the defaults now
    # print("found config")
    st_title  = config['streamlit'].get('title') or st_title
    st_header = config['streamlit'].get('header') or st_header
    st_user_input = config['streamlit'].get('user_input') or st_user_input 
    st_get_response = config['streamlit'].get('get_response') or st_get_response
    st_chbt_response =  config['streamlit'].get('chbt_response') or st_chbt_response
    st_chbt_history =  config['streamlit'].get('chbt_history') or st_chbt_history
    st_history_input  = config['streamlit'].get('history_input') or st_history_input
    st_history_output = config['streamlit'].get('history_output') or st_history_output
    st_error_text = config['streamlit'].get('error_text') or st_error_text


st.title(st_title)
st.header(st_header)
user_input = st.text_input(st_user_input)


rag_llm = load_rag_llm(config=config, secrets=secrets)

if st.button(st_get_response):
    if user_input:
        response = rag_llm.run_query(user_input)

        # Display the response
        st.markdown(st_chbt_response)
        st.success(response)

        # Optionally, you can show chat history
        st.markdown(st_chbt_history)
        st.markdown(f"{st_history_input}: {user_input}")
        st.markdown(f"{st_history_output}: {response}")
    else:
        st.error(st_error_text)
