from collections import defaultdict
import json
import time
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
import streamlit as st
import os
from utils.query_utils import load_db, query_chatbot
from utils.db_utils import add_data_to_db, data_to_db, add_uploaded_files_to_db, uploaded_files_to_db
from utils.doc_utils import uploaded_files_to_doc
from utils.evaluation_utils import evaluate_bertscore
from utils.generate_qna_utils import load_contents, generate_qna, process_text
from utils.finetune_utils import create_finetuning_job
from langchain_core.documents import Document
import shutil
import uuid
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
import plotly.express as px
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage



def db_error_check(session_state):
    if not session_state["uploaded_files"]:
        st.error("No files uploaded")
        return False
    elif not session_state["db_project"]:
        st.error("Project not specified")
        return False
    elif not session_state["openai_api_key"]:
        st.error("Please enter an OpenAI API key")
        return False
    
    return True

def visualise_bertscores(bertscores_dict):
    p_data = bertscores_dict["p"]["raw"]
    r_data = bertscores_dict["r"]["raw"]
    f1_data = bertscores_dict["f1"]["raw"]

    df = [[d, "Precision"] for d in p_data] + [[d, "Recall"] for d in r_data] + [[d, "F1 score"] for d in f1_data]

    fig = px.box(df,x = 1, y = 0)
    
    st.plotly_chart(fig, use_container_width=True)

def print_bertscores(bertscores_dict):
    for score_type in ["p", "r", "f1"]:

        
        if score_type == "p":
            score_type_long = "Precision"
        elif score_type == "r":
            score_type_long = "Recall"
        else:
            score_type_long = "F1 score"

        st.write(f"{score_type_long}: Mean = {bertscores_dict[score_type]["mean"]:.2f} (std = {bertscores_dict[score_type]["std"]:.2f})")

        # st.write(f"Mean: {bertscores_dict[score_type]["mean"]:.2f}")
        # st.write(f"Std: {bertscores_dict[score_type]["std"]:.2f}")
        # st.write(f"Max: {bertscores_dict[score_type]["max"]:.2f}")
        # st.write(f"Min: {bertscores_dict[score_type]["min"]:.2f}")
        # st.write("\n\n")

def generate_qna_streamlit(contents_directory):

    if not os.path.exists(contents_directory):
        st.error("Given directory does not exist")
        st.stop()

    contents = load_contents(contents_directory)

    llm = ChatOpenAI(model = "gpt-4o-mini")

    qnas = generate_qna(contents, llm)
    qa_pairs = process_text(qnas)
    return qa_pairs


def rescan_projects(session_state):
    available_projects = []

    # make dbs directory if it DNE
    if not os.path.exists("dbs"):
        os.makedirs("dbs", exist_ok=True)

    for file_name in os.listdir("dbs"):
        if file_name not in [".DS_Store"]:
            available_projects.append(file_name)
    
    session_state["available_projects"] = tuple(available_projects)


def write_uploaded_files_to_disk(uploaded_files, dir):

    os.makedirs(dir)

    for uploaded_file in uploaded_files:
        file_name = uploaded_file.name
        file_value = uploaded_file.getvalue()

        with open(f"{dir}/{file_name}", "wb") as f:
            f.write(file_value)

            print(f"DEBUG: writing to {dir}/{file_name}")

def cleanup_uploaded_files(dir):
    shutil.rmtree(dir)


def build_update_base(database_operation):
    if database_operation not in ["Build new", "Update existing"]:
        raise ValueError("Invalid database operation")

    if database_operation == "Update existing":
        st.header("Update")
    else:
        st.header("Build")

    st.session_state["db_type"] = database_operation

    # st.session_state["db_data_path"] = st.text_input("Path to new data directory")
    st.session_state["uploaded_files"] = st.file_uploader("Upload files to build/update database", accept_multiple_files = True)

    if st.session_state["db_type"] == "Update existing":
        st.session_state["db_project"] = st.selectbox("Select project", st.session_state["available_projects"], key = "1")

        if st.button("Rescan projects", type = "primary", key = "2"):
            rescan_projects(st.session_state)
            st.rerun()
        
        
    else:
        st.session_state["db_project"] = st.text_input("Name of project")





    root_dir = f"dbs/{st.session_state["db_project"]}"

    if st.session_state["db_type"] == "Update existing":
        st.write(f"NOTE: Existing database must be at`{root_dir}/db`")
    else:
        st.write(f"NOTE: Writing database to `{root_dir}/db`")


    if st.button("Go!"):

        if db_error_check(st.session_state):

            # write the uploaded files to temporary storage
            temp_data_dir = "f8d0ca0"
            if os.path.exists(temp_data_dir):
                cleanup_uploaded_files(temp_data_dir)
            write_uploaded_files_to_disk(st.session_state["uploaded_files"], temp_data_dir)


            if st.session_state["db_type"] == "Update existing":
                with st.spinner("Updating database..."):
                    # add_uploaded_files_to_db(
                    #     db_dir = f"{root_dir}/db",
                    #     embedding_function = OpenAIEmbeddings(),
                    #     uploaded_files = st.session_state["uploaded_files"],
                    #     llm = ChatOpenAI(model="gpt-4o-mini")
                    # )
                    add_data_to_db(
                        db_dir = f"{root_dir}/db",
                        embedding_function = OpenAIEmbeddings(),
                        new_data_directory= temp_data_dir,
                        llm = ChatOpenAI(model="gpt-4o-mini")
                    )
                st.success("Database updated!")
            else:
                with st.spinner("Creating database..."):
                    # uploaded_files_to_db(
                    #     uploaded_files = st.session_state["uploaded_files"],
                    #     embedding_function = OpenAIEmbeddings(), 
                    #     llm = ChatOpenAI(model="gpt-4o-mini"), 
                    #     save_dir = f"{root_dir}/db"
                    # )

                    data_to_db(
                        new_data_directory=temp_data_dir,
                        embedding_function = OpenAIEmbeddings(), 
                        llm = ChatOpenAI(model="gpt-4o-mini"), 
                        save_dir = f"{root_dir}/db"
                    )

                st.success("Database created!")

            # clean up temp storage
            cleanup_uploaded_files(temp_data_dir)

def title_func():
    st.title("gprMax chatbot")

    # os.environ["OPENAI_API_KEY"] = st.session_state["openai_api_key"]


def intro_func():
    st.write("Please select a function from the dropdown menu on the left")


def build_func():
    build_update_base("Build new")
        

def update_func():
    build_update_base("Update existing")




def eval_func():
    st.header("Evaluate")

    st.session_state["query_project"] = st.selectbox("Select project", st.session_state["available_projects"], key = "4")
    if st.button("Rescan projects", type = "primary", key = "7"):
        rescan_projects(st.session_state)
        st.rerun()
    # st.session_state["eval_data_path"] = st.text_input("Specify path to directory holding evaluation data. The chatbot will be evaluated on its performance on these documents. It is recommended to evaluate on the documents the chatbot was trained on.")
    st.session_state["eval_uploaded_files"] = st.file_uploader("Upload files to evaluate database", accept_multiple_files = True)
    st.session_state["eval_number"] = st.text_input("How many evaluation data points to use. Higher yields more accurate results but will take longer and use more API requests. Leave blank to use all available data")

    if st.button("Go!", key = "5"):
        if st.session_state["openai_api_key"] == "":
            st.error("Please enter an OpenAI API key")
            st.stop()
        elif st.session_state["query_project"] == "":
            st.error("Please enter a project")
            st.stop()
        elif not os.path.exists(f"dbs/{st.session_state['query_project']}/db"):
            st.error(f"Could not find database (expected at `dbs/{st.session_state['query_project']}/db`)")
            st.stop()
        
        if not os.path.exists(f"dbs/{st.session_state['query_project']}/db/chroma_db") or not os.path.exists(f"dbs/{st.session_state['query_project']}/db/docstore.pkl") or not os.path.exists(f"dbs/{st.session_state['query_project']}/db/document_data.pkl"):
            st.error("Database missing files:")
            if not os.path.exists(f"dbs/{st.session_state['query_project']}/db/chroma_db"):
                st.error(f"missing directory `dbs/{st.session_state['query_project']}/db/chroma_db`")
            
            if not os.path.exists(f"dbs/{st.session_state['query_project']}/db/docstore.pkl"):
                st.error(f"missing file `dbs/{st.session_state['query_project']}/db/docstore.pkl`")
            
            if not os.path.exists(f"dbs/{st.session_state['query_project']}/db/document_data.pkl"):
                st.error(f"missing file `dbs/{st.session_state['query_project']}/db/document_data.pkl`")

            st.stop()
        
        

        with st.spinner("Generating evaluation dataset"):

            temp_eval_data_dir = "1def5f1b"
            if os.path.exists(temp_eval_data_dir):
                cleanup_uploaded_files(temp_eval_data_dir)
            write_uploaded_files_to_disk(st.session_state["eval_uploaded_files"], temp_eval_data_dir)

            embedding_function = OpenAIEmbeddings()
            llm = ChatOpenAI(model = "gpt-4o-mini")

            qa_pairs = generate_qna_streamlit(temp_eval_data_dir)
            cleanup_uploaded_files(temp_eval_data_dir)


        with st.spinner("Evaluating"):

            embedding_function = OpenAIEmbeddings()
            llm = ChatOpenAI(model = "gpt-4o-mini")

            db, docstore = load_db(f"dbs/{st.session_state['query_project']}/db", embedding_function)

            if st.session_state["eval_number"] == "":
                n = len(qa_pairs)
            else:
                n = int(st.session_state["eval_number"])

            bertscores_dict = evaluate_bertscore(db, docstore, llm, qa_pairs = qa_pairs, n = n)

            # print_bertscores(bertscores_dict)
            visualise_bertscores(bertscores_dict)



def query_func():
    # os.environ["OPENAI_API_KEY"] = st.session_state["openai_api_key"]
    st.header("Query")



    # st.session_state["project"] = st.text_input("Project here")
    st.session_state["query_project"] = st.selectbox("Select project", st.session_state["available_projects"], key = "3")
    if st.button("Rescan projects", type = "primary", key = "6"):
        rescan_projects(st.session_state)
        st.rerun()

    st.session_state["query"] = st.text_input("Query here")

    # if st.button("Display"):
    #     st.write("openai_api_key" in st.session_state)
    #     st.write("project" in st.session_state)
    #     st.write("query" in st.session_state)

    if st.button("Go!", key = "8"):
        if st.session_state["openai_api_key"] == "":
            st.error("Please enter an OpenAI API key")
            st.stop()
        elif st.session_state["query_project"] == "":
            st.error("Please enter a project")
            st.stop()
        elif st.session_state["query"] == "":
            st.error("Please enter a query")
            st.stop()
        

        with st.spinner("Running"):

            embedding_function = OpenAIEmbeddings()
            llm = ChatOpenAI(model = "gpt-4o-mini")


            root_dir = f"dbs/{st.session_state["query_project"]}"
            db_dir = f"{root_dir}/db"

            # check if db exists
            if not os.path.exists(root_dir):
                st.error(f"Project does not exist, or is in the incorrect location. Make sure that the project exists and has path `{root_dir}`")
                st.stop()
            elif not os.path.exists(db_dir):
                st.error(f"Database does not exist, or is in the incorrect location. Make sure that the database exists and has path `{db_dir}`")
                st.stop()

            answer_path = f"{root_dir}/answers/{str(uuid.uuid4())}.md"
            os.makedirs(f"{root_dir}/answers", exist_ok=True)

            db, docstore = load_db(db_dir, embedding_function)

            try:
                answer, sources = query_chatbot(st.session_state["query"], db, docstore, llm)
            except:
                st.error("Unable to generate answer. Please check OpenAI API key, or try again later")
                st.stop()

            st.header("Answer:")
            st.markdown(answer)

            st.header("Sources:")
            for s in sources:
                st.write(s)



def chat_func():

    title_func()

    AI_AVATAR = "images/gprMax_FB_logo.png"

    # os.environ["OPENAI_API_KEY"] = st.session_state["openai_api_key"]
    # st.session_state["project"] = st.text_input("Project here")
    with st.sidebar:

        chat_model = st.selectbox("Select a model", ["gpt-4o-mini", "gpt-4o"])

        st.session_state["query_project"] = "af6c69d5"

    for msg in history.messages:
        avatar = AI_AVATAR if msg.type == "ai" else None
        st.chat_message(msg.type, avatar=avatar).write(msg.content)

    if len(history.messages) == 0:
        with st.chat_message("assistant", avatar=AI_AVATAR):
            st.write("Ask away!")
        history.add_ai_message("Ask away")

    streamlit_prompt = st.chat_input("What would you like to know?")


    if streamlit_prompt:
        if st.session_state["openai_api_key"] == "":
            st.error("Please enter an OpenAI API key")
            st.stop()
        elif not chat_model:
            st.error("Please specify a chat model")
            st.stop()
        elif st.session_state["query_project"] == "":
            st.error("Please enter a project")
            st.stop()
        elif streamlit_prompt == "":
            st.error("Please enter a query")
            st.stop()

        with st.chat_message("user"):
            st.write(streamlit_prompt)
        history.add_user_message(streamlit_prompt)
        

        embedding_function = OpenAIEmbeddings()
        llm = ChatOpenAI(model = chat_model)


        root_dir = f"dbs/{st.session_state["query_project"]}"
        db_dir = f"{root_dir}/db"

        # check if db exists
        if not os.path.exists(root_dir):
            st.error(f"Project does not exist, or is in the incorrect location. Make sure that the project exists and has path `{root_dir}`")
            st.stop()
        elif not os.path.exists(db_dir):
            st.error(f"Database does not exist, or is in the incorrect location. Make sure that the database exists and has path `{db_dir}`")
            st.stop()

        answer_path = f"{root_dir}/answers/{str(uuid.uuid4())}.md"
        os.makedirs(f"{root_dir}/answers", exist_ok=True)

        db, docstore = load_db(db_dir, embedding_function)

        answer, sources = query_chatbot(streamlit_prompt, db, docstore, llm)

        with st.chat_message("assistant", avatar=AI_AVATAR):
            st.write(answer)
        history.add_ai_message(answer)

        with st.expander("See sources"):
            for s in set(sources):
                st.write(f"- {s}")

    

def finetune_func():
    st.header("Finetune")

    st.write("See [here](https://platform.openai.com/docs/guides/fine-tuning/preparing-your-dataset) for how to prepare data for finetuning.")

    # st.session_state["db_data_path"] = st.text_input("Path to new data directory")
    train_file = st.file_uploader("Upload training files for finetuning (required)", accept_multiple_files = False)
    val_file = st.file_uploader("Upload validation files for finetuning (leave blank if no validation file)", accept_multiple_files = False)

    finetune_base_model = st.selectbox("Select base model", ["gpt-4o-mini-2024-07-18", "gpt-4o-2024-08-06", "gpt-3.5-turbo-1106", "gpt-3.5-turbo-0613", "gpt-3.5-turbo-0125", "davinci-002", "babbage-002"])


    if st.button("Go!"):

        if not train_file:
            st.error("Please supply a training file")
            st.stop()

        # write the uploaded files to temporary storage
        temp_data_dir = "ft8d0ca0"
        if os.path.exists(temp_data_dir):
            cleanup_uploaded_files(temp_data_dir)
        if val_file:
            write_uploaded_files_to_disk([train_file, val_file], temp_data_dir)
        else:
            write_uploaded_files_to_disk([train_file], temp_data_dir)

        # validate format
        if not streamlit_validate_ft_format(f"{temp_data_dir}/{train_file.name}"):
            st.error("Training file is not in a valid format. See [here](https://platform.openai.com/docs/guides/fine-tuning/preparing-your-dataset) for proper format.")
            st.stop()
        
        if val_file and not streamlit_validate_ft_format(f"{temp_data_dir}/{val_file.name}"):
            st.error("Validation file is not in a valid format. See [here](https://platform.openai.com/docs/guides/fine-tuning/preparing-your-dataset) for proper format.")
            st.stop()


        with st.spinner("Creating finetuning job..."):

            create_finetuning_job(
                train_path=f"{temp_data_dir}/{train_file.name}",
                val_path=f"{temp_data_dir}/{val_file.name}" if val_file else None,
                model=finetune_base_model
            )

        time.sleep(3)

        # clean up temp storage
        cleanup_uploaded_files(temp_data_dir)

        st.success("Finetuning job created! Check [OpenAI finetuning dashboard](https://platform.openai.com/finetune/ftjob-Waatof7eHXARVQoxwvHAGrEH?filter=all) to see status")

def streamlit_validate_ft_format(data_path):
    '''
    source: https://cookbook.openai.com/examples/chat_finetuning_data_prep
    '''

    def load_data(data_path):
        with open(data_path, 'r', encoding='utf-8') as f:
            dataset = [json.loads(line) for line in f]

        return dataset

    try:
        dataset = load_data(data_path)
    except:
        st.error("Error loading dataset. See [here](https://platform.openai.com/docs/guides/fine-tuning/preparing-your-dataset) for proper format.")
        st.stop()

    format_errors = defaultdict(int)

    for ex in dataset:
        if not isinstance(ex, dict):
            format_errors["data_type"] += 1
            continue
            
        messages = ex.get("messages", None)
        if not messages:
            format_errors["missing_messages_list"] += 1
            continue
            
        for message in messages:
            if "role" not in message or "content" not in message:
                format_errors["message_missing_key"] += 1
            
            if any(k not in ("role", "content", "name", "function_call", "weight") for k in message):
                format_errors["message_unrecognized_key"] += 1
            
            if message.get("role", None) not in ("system", "user", "assistant", "function"):
                format_errors["unrecognized_role"] += 1
                
            content = message.get("content", None)
            function_call = message.get("function_call", None)
            
            if (not content and not function_call) or not isinstance(content, str):
                format_errors["missing_content"] += 1
        
        if not any(message.get("role", None) == "assistant" for message in messages):
            format_errors["example_missing_assistant_message"] += 1

    if format_errors:
        st.error("Found errors:")
        for k, v in format_errors.items():
            st.error(f"{k}: {v}")
        return False
    else:
        # print("No errors found")
        return True

def input_func():
    st.title("Input file generation")

    AI_AVATAR = "images/gprMax_FB_logo.png"

    # os.environ["OPENAI_API_KEY"] = st.session_state["openai_api_key"]
    # st.session_state["project"] = st.text_input("Project here")

    chat_model = "ft:gpt-4o-mini-2024-07-18:personal::A3Xj55I8"

    st.session_state["query_project"] = "af6c69d5"

    for msg in history.messages:
        avatar = AI_AVATAR if msg.type == "ai" else None
        st.chat_message(msg.type, avatar=avatar).write(msg.content)

    if len(history.messages) == 0:
        with st.chat_message("assistant", avatar=AI_AVATAR):
            st.write("Ask away!")
        history.add_ai_message("Ask away")

    streamlit_prompt = st.chat_input("Request input file for generation")


    if streamlit_prompt:
        if st.session_state["openai_api_key"] == "":
            st.error("Please enter an OpenAI API key")
            st.stop()
        elif not chat_model:
            st.error("Please specify a chat model")
            st.stop()
        elif st.session_state["query_project"] == "":
            st.error("Please enter a project")
            st.stop()
        elif streamlit_prompt == "":
            st.error("Please enter a query")
            st.stop()

        with st.chat_message("user"):
            st.write(streamlit_prompt)
        history.add_user_message(streamlit_prompt)
        

        embedding_function = OpenAIEmbeddings()
        llm = ChatOpenAI(model = chat_model)


        root_dir = f"dbs/{st.session_state["query_project"]}"
        db_dir = f"{root_dir}/db"

        # check if db exists
        if not os.path.exists(root_dir):
            st.error(f"Project does not exist, or is in the incorrect location. Make sure that the project exists and has path `{root_dir}`")
            st.stop()
        elif not os.path.exists(db_dir):
            st.error(f"Database does not exist, or is in the incorrect location. Make sure that the database exists and has path `{db_dir}`")
            st.stop()

        answer_path = f"{root_dir}/answers/{str(uuid.uuid4())}.md"
        os.makedirs(f"{root_dir}/answers", exist_ok=True)

        db, docstore = load_db(db_dir, embedding_function)

        answer, sources = query_chatbot_input(streamlit_prompt, db, docstore, llm)

        # reformat "answer" to have spaces before each "#" delimiter
        answer = answer.replace("#", "\n#")

        with st.chat_message("assistant", avatar=AI_AVATAR):
            st.write(answer)
        history.add_ai_message(answer)

        with st.expander("See sources"):
            for s in set(sources):
                st.write(f"- {s}")


def query_chatbot_input(query, db, docstore, llm):

    # get relevant summaries
    query_docs = db.similarity_search(query)

    # get original documents
    source_docs = []
    for doc in query_docs:
        source_docs.append(docstore[doc.metadata['doc_id']])

    prompt = "You are an assistant for generating input files for gprMax. Use the following pieces of retrieved context to generate an input file for the question. If you don't know the answer, just say that you don't know.\
    Question: {question}\
    Context: {context}\
    Answer:"

    context = " ".join([doc.page_content for doc in source_docs])


    prompt_template = PromptTemplate.from_template(prompt)


    messages = [
        SystemMessage(
            content = "Write the gprMax input file for this"
        ),
        HumanMessage(
            content = prompt_template.format(question = query, context = context)
        )
    ]

    # print(f"DEBUG: {messages}")

    chain = (llm | StrOutputParser())

    answer = chain.invoke(messages)

    sources = [doc.metadata['source'] for doc in source_docs]

    return answer, sources
    

history = StreamlitChatMessageHistory(key = "chat_messages")