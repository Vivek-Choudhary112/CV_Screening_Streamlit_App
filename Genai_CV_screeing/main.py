import streamlit as st
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
from langchain.chains import RetrievalQA
from langchain.llms import GooglePalm
from PIL import Image
from InstructorEmbedding import INSTRUCTOR
from langchain.embeddings import HuggingFaceInstructEmbeddings
import pdb
import subprocess
import ctypes
load_dotenv()
# os.getenv("GOOGLE_API_KEY")
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))



def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    # embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    # # Initialize instructor embeddings using the Hugging Face model
    instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vector_store = FAISS.from_texts(text_chunks, embedding=instructor_embeddings)
    vector_store.save_local("faiss_index")




# def get_conversational_chain():
#     prompt_template = """
#     Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
#     provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
#     Context:\n {context}?\n
#     Question: \n{question}\n
#
#     Answer:
#     """
#
#     model = ChatGoogleGenerativeAI(model="gemini-pro",
#                                    temperature=0.3)
#
#     prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
#     chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
#
#     return chain


def get_conversational_chain(new_db, llm):
    prompt_template = """You are a master in CV screening, who specializes in CV screening comparing the values with the context only. 
       Please provide all information from the context only.
       If any proper noun as question provided matches the context:
           Respond with detailed information as much possible from the provided {context}. Don't try to make up an result.
     
       If any proper noun as question provided doesn't matches the context, Just say, "{question} is not available in the any pdf files".  Don't try to make up an result.
    
       Given the following context and a question from user, generate an result based on this context only, don't try to make up an result.

           CONTEXT: {context}

           QUESTION: {question}
    """

    # model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    retriever = new_db.as_retriever(score_threshold=1)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain_type_kwargs = {"prompt": prompt}
    chain = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type="stuff",
                                        retriever=retriever,
                                        input_key="question",
                                        return_source_documents=True,
                                        chain_type_kwargs=chain_type_kwargs)

    return chain


def user_input(user_question):
    llm = GooglePalm(google_api_key=os.environ["GOOGLE_API_KEY"], temperature=0)
    # embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain(new_db, llm)

    response = chain(
        {"input_documents": docs, "question": user_question}
        , return_only_outputs=True)

    print(response)
    st.write("Reply: ", response["result"])


def remove_with_elevated_permissions(path):
   if os.path.exists(path):
       try:
           # Specify the path to the file you want to remove
           # Call the Windows API function DeleteFileW with the file path
           # if ctypes.windll.kernel32.DeleteFileW(path):
           #     return("File deleted successfully.")
           # else:
           #     return("Failed to delete the file.")

           if os.path.isfile(path):
               subprocess.run(["sudo", "rm", path], check=True)
               subprocess.run(["runas", "/user:Administrator", "del", path])
               return (f"File '{path}' has been successfully deleted with elevated permissions.")
           elif os.path.isdir(path):
               # subprocess.call('dir', shell=True)
               # subprocess.call(['cmd', '/c', 'dir'])
               # subprocess.run(["sudo", "rm", "-r", path], shell=True)
               subprocess.run(["runas", "/user:Administrator", "rmdir", "/s", "/q", path])
               os.remove("faiss_index")
               # subprocess.call(['cmd', '/c', 'dir'])
               return (f"Folder '{path}' has been successfully deleted with elevated permissions.")
       except subprocess.CalledProcessError as e:
           return (f"Error: {e}")
   else:
       return (f"The specified path '{path}' does not exist.")


def main():
    path = "/mount/src/cv_screening_streamlit_app/Genai_CV_screeing/faiss_index"
    st.set_page_config("EY CV screening App")
    image = Image.open("/mount/src/cv_screening_streamlit_app/Genai_CV_screeing/ey logo.jpg")
    # image_resized = image.resize((,100))
    st.image(image, use_column_width=True)
    st.header("CV screening 🕵")
    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button",
                                    accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                # pdb.set_trace()
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")
        if st.button("Exit"):
            with st.spinner("Exiting..."):
                if os.path.exists("faiss_index"):
                    print(True)
                    file_remove = remove_with_elevated_permissions("faiss_index")
                    # os.remove("faiss_index")
                    st.success(file_remove)
                    st.success("Embeddings successfully removed")
                else:
                    st.warning("Embeddings doesn't exists")


if __name__ == "__main__":
    main()
