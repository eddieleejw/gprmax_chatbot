import streamlit as st
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
import os
import uuid
from utils.query_utils import load_db, query_chatbot
from utils.db_utils import add_data_to_db, data_to_db, add_uploaded_files_to_db, uploaded_files_to_db
from utils.streamlit_utils import cleanup_uploaded_files, db_error_check, print_bertscores, generate_qna_streamlit, rescan_projects, write_uploaded_files_to_disk, build_func, update_func, eval_func, query_func, title_func, intro_func, chat_func, finetune_func, input_func
from utils.evaluation_utils import evaluate_bertscore
import uuid


# read in available projects once
if "available_projects" not in st.session_state:
    rescan_projects(st.session_state)


if "openai_api_key" not in st.session_state:
    title_func()
    open_ai_api_key = st.text_input("OpenAI API key here", key = "10")
    if st.button("Enter"):
        st.session_state["openai_api_key"] = open_ai_api_key
        st.rerun()
    else:
        st.stop()



os.environ["OPENAI_API_KEY"] = st.session_state["openai_api_key"]


page_names_to_funcs = {
    "Chat": chat_func,
    "Input files": input_func
}
page_name = st.sidebar.selectbox("Mode", page_names_to_funcs.keys())

page_names_to_funcs[page_name]()