import streamlit as st
import time
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_google_vertexai import ChatVertexAI
from langchain_google_vertexai import VertexAI
from vertexai import init
from langchain_google_vertexai import VertexAI
from langchain_google_vertexai.embeddings import VertexAIEmbeddings
from streamlit_feedback import streamlit_feedback
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.memory import ConversationBufferMemory, StreamlitChatMessageHistory
from langchain.callbacks.tracers.run_collector import RunCollectorCallbackHandler
from langchain.schema.runnable import RunnableConfig


PROJECT_ID = "endless-radar-434900-j9"
REGION = "europe-west2"
MODEL_ID = "text-embedding-004"

import vertexai
from vertexai.language_models import TextEmbeddingModel

vertexai.init(project=PROJECT_ID, location=REGION)

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = "AIzaSyDm4bCaEZLJ-NhJejSRyF1iOq2V2jXLlW8"
#AIzaSyDm4bCaEZLJ-NhJejSRyF1iOq2V2jXLlW8


def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text

run_collector = RunCollectorCallbackHandler()
runnable_config = RunnableConfig(
    callbacks=[run_collector],
    tags=["Streamlit Chat"],
)

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    return chunks

# def ingestion():
#     loader = PyPDFLoader("./rag-vertex/learn_ebpf_setup.pdf")
#     documents = loader.load_and_split()

#     embedding_model= VertexAIEmbeddings("text-embedding-004")
    
#     vectorstore = FAISS.from_documents(documents=documents,embedding=embedding_model)

#     vectorstore.save_local("faiss_index_vectorstore")




def get_vector_store(text_chunks):
    embedding_model= VertexAIEmbeddings("text-embedding-004")
    vector_store = FAISS.from_texts(text_chunks, embedding=embedding_model)
    vector_store.save_local("faiss_index")
    


def get_conversational_chain():

    prompt_template_rag = """You are costomer service assistance you have to help clients with their needs. Make sure to greet them as first message but only it was first message
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    chat_history:\n{chat_history}\n

    Answer:
    """

    prompt_template = """You are costomer service assistance you have to help clients with their needs.
    Answer the question as detailed as possible make sure to provide all the details, if you don't know the answer says, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    chat_history:\n{chat_history}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro-002",
                             temperature=0.3)

    prompt_rag = PromptTemplate(template = prompt_template_rag, input_variables = ["context", "question","chat_history"])
    chain_rag= load_qa_chain(model, chain_type="stuff", prompt=prompt_rag)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question","chat_history"])
    chain= load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain_rag,chain

    def firestro():
        #TODO: add storage
        pass




def user_input_rag(user_question,chat_history):
    embeddings = VertexAIEmbeddings("text-embedding-004")
    
    new_db = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain_rag,chain = get_conversational_chain()

    
    response = chain_rag(
        {"input_documents":docs, "question": user_question,"chat_history":chat_history}
        , return_only_outputs=True)

    return response,docs
    #st.write("Reply: ", response["output_text"])


def user_input(user_question,chat_history):
    #embeddings = VertexAIEmbeddings("text-embedding-004")
    
    #new_db = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
    docs = ""

    chain_rag,chain = get_conversational_chain()

    
    response = chain(
        {"input_documents":docs, "question": user_question,"chat_history":chat_history}
        , return_only_outputs=True)

    return response,docs


import streamlit as st

st.title("GCP chat")


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "run_id" not in st.session_state:
    st.session_state.run_id = None

memory = ConversationBufferMemory(
    chat_memory=StreamlitChatMessageHistory(key="langchain_messages"),
    return_messages=True,
    memory_key="chat_history",
)

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if prompt := st.chat_input("What is up?"):
    # Create two columns
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    col1, col2 = st.columns(2)
    with st.spinner("waiting"):
        response1,docs1 = user_input(user_question=prompt,chat_history=st.session_state.messages)
        response2,docs2 = user_input_rag(user_question=prompt,chat_history=st.session_state.messages)


    # Display output in the first column
    with col1:
        st.markdown("Model Output without RAG")
        with st.chat_message("assistant"):
            #with st.spinner("waiting"):
            #response,docs = user_input(user_question=prompt,chat_history=st.session_state.messages)
            words = response1["output_text"].split()

            # Create a placeholder for the word-by-word output
            placeholder = st.empty()

            current_text = ""
            
            # Display each word with a slight delay
            for word in words:
                current_text += word + " "  # Update the text with the next word
                placeholder.markdown(current_text)  # Display the updated text
                time.sleep(0.1)  # Adjust delay as needed
            run = run_collector.traced_runs[0]
            run_collector.traced_runs = []
            st.session_state.run_id = run.id
            wait_for_all_tracers()
            # Final display with complete response
            placeholder.markdown(current_text.strip())
            if st.session_state.get("run_id"):
                feedback = streamlit_feedback(
                    feedback_type="faces",  # Apply the selected feedback style
                    optional_text_label="[Optional] Please provide an explanation",  # Allow for additional comments
                    key=f"feedback_{st.session_state.run_id}",
                )
        st.session_state.messages.append({"role": "assistant_norag", "content": response1["output_text"],"chuncks":docs1})
    # Display output in the second column
    with col2:
        st.markdown("Model Output with RAG")
        with st.chat_message("assistant"):
            #with st.spinner("waiting"):
            #response,docs = user_input_rag(user_question=prompt,chat_history=st.session_state.messages)
            words = response2["output_text"].split()

            # Create a placeholder for the word-by-word output
            placeholder = st.empty()

            current_text = ""
            
            # Display each word with a slight delay
            for word in words:
                current_text += word + " "  # Update the text with the next word
                placeholder.markdown(current_text)  # Display the updated text
                time.sleep(0.1)  # Adjust delay as needed
            run = run_collector.traced_runs[0]
            run_collector.traced_runs = []
            st.session_state.run_id = run.id
            wait_for_all_tracers()
            # Final display with complete response
            placeholder.markdown(current_text.strip())
            # st.markdown(response["output_text"])
            # for doc in docs:
            #     st.markdown(doc)
            if st.session_state.get("run_id"):
                feedback = streamlit_feedback(
                    feedback_type="faces",  # Apply the selected feedback style
                    optional_text_label="[Optional] Please provide an explanation",  # Allow for additional comments
                    key=f"feedback_{st.session_state.run_id}",
                )
        st.session_state.messages.append({"role": "assistant_rag", "content": response2["output_text"],"chuncks":docs2})
# React to user input
#if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    # st.chat_message("user").markdown(prompt)
    # # Add user message to chat history
    # st.session_state.messages.append({"role": "user", "content": prompt})


    # Display assistant response in chat message container
        # with st.chat_message("assistant"):
        #     with st.spinner("waiting"):
        #         response,docs = user_input(user_question=prompt,chat_history=st.session_state.messages)
        #         st.markdown(response["output_text"])
        #         st.write("Docs: ", docs)
                # for doc in docs:
                #     with st.expander(doc["title"]):
                #         st.write(doc["content"])
                
        # Add assistant response to chat history
        # st.session_state.messages.append({"role": "assistant", "content": response["output_text"],"chuncks":docs})

with st.sidebar:
    st.title("Menu:")
    pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
    if st.button("Submit & Process"):
        with st.spinner("Processing..."):
            raw_text = get_pdf_text(pdf_docs)
            text_chunks = get_text_chunks(raw_text)
            get_vector_store(text_chunks)
            st.success("Done")


def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using Gemini💁")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    with st.chat_message("user"):
        st.write("Hello 👋")

        user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

# if __name__ == "__main__":
# #    main()
