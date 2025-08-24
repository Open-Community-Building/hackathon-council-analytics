#streamlit run src/web_app.py ~/.config/hca/ --server.port=8686  --server.address=0.0.0.0
import streamlit as st
import sys
import os
from ragllm import RagLlm
import tomllib
import toml

# Define Defaults
configdir = os.path.expanduser('~') + "/.config/hca/"

print(f"Using config directory: {configdir}")
print(f"Systemargumente: {sys.argv}")
# FIXME: Start with correct config path from command line instead of Docker path

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

print(f"Using config directory: {configdir}")
config = read_config(os.path.join(configdir, 'config.toml'))
secrets = read_config(os.path.join(configdir, 'secrets.toml'))



if config and config.get('streamlit'):
    st_title  = config['streamlit'].get('title') or st_title
    st_header = config['streamlit'].get('header') or st_header
    st_user_input = config['streamlit'].get('user_input') or st_user_input 
    st_get_response = config['streamlit'].get('get_response') or st_get_response
    st_chbt_response =  config['streamlit'].get('chbt_response') or st_chbt_response
    st_chbt_history =  config['streamlit'].get('chbt_history') or st_chbt_history
    st_history_input  = config['streamlit'].get('history_input') or st_history_input
    st_history_output = config['streamlit'].get('history_output') or st_history_output
    st_error_text = config['streamlit'].get('error_text') or st_error_text


page = st.sidebar.radio("Seite auswählen", ["Chat", "Konfiguration"])

if page == "Konfiguration":
    st.title("Konfiguration")
    
    if "frameworks" not in config:
        config["frameworks"] = {
            "filestorage": "local",
            "embed": config["model"]["embed_name"],
            "llm": config["model"]["llm_name"]
        }
    
    available_filestorages = [key for key, value in config.get("documents", {}).items() if isinstance(value, dict)]
    selected_filestorage = st.selectbox(
        "Wähle den Filestorage-Typ",
        available_filestorages,
        index=available_filestorages.index(config["documents"]["filestorage"])
    )

    available_embeddings = [key for key, value in config.get("embedding", {}).items() if isinstance(value, dict)]
    selected_embedding = st.selectbox(
        "Wähle das Embedding-Modell",
        available_embeddings,
        index=available_embeddings.index("faiss")
    )    

    # available_llms = [key for key, value in config.get("embedding", {}).items() if isinstance(value, dict)]
    # selected_llm = st.selectbox(
    #     "Wähle das LLM-Modell",
    #     llm_options,
    #     index=llm_options.index(config["model"]["llm_name"])
    # )
    
    config["model"]["filestorage"] = selected_filestorage
    config["model"]["embed_name"] = selected_embedding
    # config["model"]["llm_name"] = selected_llm
    
    st.write("Aktuelle Modell-Konfiguration:", config["model"])
    
    if st.button("Konfiguration speichern"):
        with open(configfile, "w") as f:
            toml.dump(config, f)
        st.success("Konfiguration wurde aktualisiert!")
    
    st.stop()


if page == "Chat":
    st.title(st_title)
    st.header(st_header)
    user_input = st.text_input(st_user_input)

    rag_llm = load_rag_llm(config=config, secrets=secrets)

    if st.button(st_get_response):
        if user_input:
            response = rag_llm.run_query(user_input)
            print(f"Response: {response}")

            st.markdown(st_chbt_response)
            st.success(response)

            st.markdown(st_chbt_history)
            st.markdown(f"{st_history_input}: {user_input}")
            st.markdown(f"{st_history_output}: {response}")
        else:
            st.error(st_error_text)
