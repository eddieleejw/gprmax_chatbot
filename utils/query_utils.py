'''
use command:

python query.py {project_name} query/prompt {query}
'''

from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
import pickle
import os

def generate_gprmax_input(query):
    """
    Generates a gprMax input file based on user query.
    """
    input_template = """#title {title}
#domain: {x} {y} {z}
#dx_dy_dz: {dx} {dy} {dz}
#waveform: {waveform}
#rx: {rx_x} {rx_y} {rx_z}
#tx: {tx_x} {tx_y} {tx_z}
"""
    
    parameters = {
        "title": "Generated Input File",
        "x": 1.0, "y": 1.0, "z": 1.0,
        "dx": 0.01, "dy": 0.01, "dz": 0.01,
        "waveform": "ricker 1 1e9",
        "rx_x": 0.5, "rx_y": 0.5, "rx_z": 0.1,
        "tx_x": 0.0, "tx_y": 0.0, "tx_z": 0.0
    }
    
    input_content = input_template.format(**parameters)
    file_path = f"generated_input_{os.getpid()}.in"
    
    with open(file_path, "w") as f:
        f.write(input_content)
    
    return file_path

def query_chatbot(query, db, docstore, llm):
    """
    Modified chatbot function to detect input file generation requests.
    """
    if "generate input file" in query.lower():
        return generate_gprmax_input(query), []

    query_docs = db.similarity_search(query)
    source_docs = [docstore[doc.metadata['doc_id']] for doc in query_docs]
    
    prompt = "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know.\
    Question: {question}\
    Context: {context}\
    Answer:"
    
    context = " ".join([doc.page_content for doc in source_docs])
    
    prompt_template = PromptTemplate.from_template(prompt)
    chain = (prompt_template | llm | StrOutputParser())
    
    answer = chain.invoke({"question": query, "context": context})
    sources = [doc.metadata['source'] for doc in source_docs]
    
    return answer, sources

def load_db(db_dir, embedding_function):
    with open(f"{db_dir}/docstore.pkl", "rb") as f:
        docstore = pickle.load(f)
    
    db = Chroma(persist_directory=f"{db_dir}/chroma_db", embedding_function=embedding_function)
    
    return db, docstore

def save_answer_and_sources(answer_path, query, answer, sources):
    with open(answer_path, "w") as f:
        f.write("# Query:\n")
        f.write(query)
        f.write("\n\n")
        
        f.write("# Answer:\n")
        f.write(answer)
        f.write("\n\n")
        
        f.write("## Sources:\n")
        for s in sources:
            f.write(f"* {s}\n")

def get_prompt(query, db, docstore):
    query_docs = db.similarity_search(query)
    source_docs = [docstore[doc.metadata['doc_id']] for doc in query_docs]
    
    prompt = "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know.\
    Question: {question}\
    Context: {context}\
    Answer:"
    
    context = " ".join([doc.page_content for doc in source_docs])
    
    prompt_template = PromptTemplate.from_template(prompt)
    prompt = prompt_template.invoke({"question": query, "context": context})
    
    sources = [doc.metadata['source'] for doc in source_docs]
    
    return prompt.text, sources

def copy_prompt(prompt, sources):
    print("Prompt copied to clipboard\n")
    print("Sources:")
    for s in sources:
        print(f"{s}")

def print_answer(answer, sources, answer_path):
    print("==========================")
    print("Answer:")
    print(answer)
    print("\n\n")
    
    print("==========================")
    print("Sources:")
    for s in sources:
        print(s)
    print("\n\n")
    
    print(f"Saved to {answer_path}")
