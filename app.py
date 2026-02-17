import streamlit as st
from pypdf import PdfReader
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline
import torch


# ------------------------------------------------------------------------------------- #
# Initialization of state:
if "pdf_text" not in st.session_state:
    st.session_state.pdf_text = ""

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "embeddings" not in st.session_state:
    st.session_state.embeddings = None

if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None

if "summary" not in st.session_state:
    st.session_state.summary = ""

#------------------------------------------------------------------------------------- # 
# TEXT EXTRACTION:
def extract_pdfs(uploaded_files):
        all_text = ""

        for pdf in uploaded_files:
                reader = PdfReader(pdf)
                for page in reader.pages:
                        text = page.extract_text()
                        if text:
                                all_text += text + "\n"
        return all_text.strip()
# ----------------------------------------------------------------------------------------- #     
# Chunking function: 
# """Splits large text into smaller chunks."""
def get_chunks(text):
      text_splitter = CharacterTextSplitter(
            separator = "\n",
            chunk_size = 1000,
            chunk_overlap = 200,
            length_function = len
      )
      chunks = text_splitter.split_text(text) 
      return chunks


# -------------------------------------------------------------------------------------------------
# Create embeddings and FAISS index
# import torch
device = "cpu"  # force CPU
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def create_faiss_index(chunks):
    if not chunks:
        return None, None
    
# convert chunsk to embeddings
    embeddings = embedding_model.encode(chunks, convert_to_numpy=True)
    
# create FAISS indec
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    return index, embeddings

# ------------------------------------------------------------------------------------------------------------ #
def search(query, top_k=5):
    if not st.session_state.faiss_index or not st.session_state.embeddings.any():
        return []
    query_vec = embedding_model.encode([query], convert_to_numpy=True)
    distances, indices = st.session_state.faiss_index.search(query_vec, top_k)
    results = [st.session_state.chunks[i] for i in indices[0]]
    return results
# ------------------------------------------------------------------------------------------------- #
# def get_vectorstore(chunks):
#       embeddings = SentenceTransformer(model_name ="sentence-transformers/all-MiniLM-L6-v2")
#       vectorstore = faiss.from_texts(texts=chunks, embeddings=embeddings)
#       return vectorstore
      

# Clean PDFs
# def clean(text):
#     text = re.sub(r"\s+", " ", text)
#     return text.strip()
# --------------------------------------------------------------------------------------------------------------------- # 
# Hugging Face summarisation pipeline
# @st.cache_resource
# def get_summarizer():
#     summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
#     return summarizer

# summarizer = get_summarizer()
# ---------------------------------------------------------------------------------------------------------------------- #


def main():
    # load_dotenv()
    # print("hello world")
    st.set_page_config(
        page_title="Mini PDF AI Assistant",
        page_icon=":books:", 
        layout="wide"
        )


# ---------------------------------------------------------------------------------------------------------------- #
# MOVED AT THE TOP:

# if "pdf_text" not in st.session_state:
#         st.session_state.pdf_text = ""

# if "chunks" not in st.session_state:
#         st.session_state.chunks = []

# if "embeddings" not in st.session_state:
#         st.session_state.embeddings = None

# if "faiss_index" not in st.session_state:
#     st.session_state.faiss_index = None

# if "summary" not in st.session_state:
#     st.session_state.summary = ""
# ----------------------------------------------------------------------------------------------------------------- #

# st.text_input("Ask a questions about the document!")

with st.sidebar:
    st.header("Upload PDFs")

    uploaded_files = st.file_uploader(
                            "Upload your pdf(s)",
                           type = "pdf",
                            accept_multiple_files=True
                            )

    # if st.button("Process"):
    #       with st.spinner("Proccessing"):
process_button = st.sidebar.button("Process PDFs")

st.sidebar.title("Mini Study Buddy!") #  AI Assistant
st.sidebar.markdown("Upload PDFs, Summarize, Chat, and Generate questions.")
# ------------------------------------------------------------------------------- #

# Main Page
st.header("Summarise and Q&A :books:")
st.divider()
# ------------------------------------------------------------------------------- # 

# # get pdf text
# raw_text = extract_pdfs(uploaded_files)
# st.write(raw_text)



# # get the text
# chunks = get_chunks(raw_text)
# st.write(chunks)


# create vector store ====> 
# vectorstore = get_vectorstore(chunks)
# st.divider()

# ------------------------------------------------------------------------------------------------------------ #
#  Summary:
st.subheader("Summary Process")


# if process_button:
#    if uploaded_files:
#                  st.info(f"Processing, please wait!")
#    else:
#                  st.warning("Upload a pdf")

# else:
#                  st.write("Upload PDFs and click 'Process PDFs'")



if process_button:
        if uploaded_files:
            with st.spinner("Extracting text from PDFs..."):
                extracted_text = extract_pdfs(uploaded_files)
                st.session_state.pdf_text = extracted_text
                st.session_state.chunks = get_chunks(extracted_text)

            if extracted_text:
                st.success("Text extracted successfully")
            else:
                st.warning("No text found in the uploaded PDFs.")


            if st.session_state.chunks:
                with st.spinner("Creating embeddings and FAISS vector database..."):
                    st.session_state.faiss_index, st.session_state.embeddings = create_faiss_index(st.session_state.chunks)
            st.success("Embeddings and FAISS index ready")


        else:
            st.warning("Upload a PDF")
else:
        st.write("Upload PDFs and click 'Process PDFs'")

st.divider()
# --------------------------------------------------------------------------------------------------------------------------------------------------- #
# DEBUG CONFIRMATION (TEMP)
# =========================
if st.session_state.pdf_text:
        st.write("✅ Text extracted")

if st.session_state.chunks:
        st.write("✅ Chunks ready")

if st.session_state.embeddings is not None:
    st.write("✅ Created embeddings")

st.divider()
             
# SUMMARISATION: --------------------------------------------------------------------------------------------------------------------------
st.subheader("📝 Summarisation")

summarise_button = st.button("Generate Summary")

if summarise_button:
    if st.session_state.chunks:
        with st.spinner("Summary is being generated..."):
            # Combine all chunks into one text
            combined_text = "\n".join(st.session_state.chunks)
            
# Use Bart - huggingface summarisation -
# -----------------------------------------------------
            # max_chunk = 1000  # characters
            # summaries = []
            # for i in range(0, len(combined_text), max_chunk):
            #     chunk = combined_text[i:i+max_chunk]
            #     summary_chunk = summarizer(chunk, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
            #     summaries.append(summary_chunk)

            # summary = " ".join(summaries)
# --------------------------------------------------------------  
            summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)  # -1 for CPU
            summary_text = summarizer(combined_text, max_length=250, min_length=80, do_sample=False)[0]['summary_text']
            # Store summary in session state
            # st.session_state.summary = summary
            st.session_state.summary = summary_text

            
            st.success("Summary generated ✅")
            # st.write(summary)
            st.write(summary_text)
            
    else:
        st.warning("No text chunks found. Please uploadna PDF.")

st.divider()


# -------------------
# DEBUG CONFIRMATION (TEMP)
# =========================
# if st.session_state.pdf_text:
#         st.write("✅ Text extracted")

# if st.session_state.chunks:
#         st.write("✅ Chunks ready")

# if st.session_state.embeddings is not None:
#     st.write("✅ Created embeddings")

# st.divider()
             

# ---------------------------------------------------------------------------------------------------------- #
# Chat 
st.subheader("💬 Chat ")
user_question = st.text_input("Ask a question about the PDFs:",
                              key="active_question_input")
ask_button = st.button("Ask", key="ask_button")
              
if ask_button:
    if st.session_state.chunks and st.session_state.embeddings is not None:
        with st.spinner("Searching for answer..."):
            results = search(user_question)
            st.write("### Relevant chunks from PDFs:")
            for i, res in enumerate(results):
                st.write(f"{i+1}. {res}")
    else:
        st.warning("Upload PDFs and process them first.")
# if ask_button:
#               if uploaded_files:
#                  st.info("Processing, please wait!")
#               else:
#                  st.warning("Upload a pdf")

# ===> 
# st.subheader("💬 Chat")
# st.text_input("Ask a question about the PDFs:", disabled=True)
st.divider()

# -------------------------------------------------------------------------------------------------- #
# Generate own Questions 
# st.subheader("❓ Generated Questions")
# generate_button = st.button("Generate Questions")

# if generate_button:
#             if uploaded_files:
#              st.info("Processing, please wait!")
# else:
#              st.warning("Upload a pdf")

# ===> 
# st.button("Generate Questions", disabled=True)

if __name__ == '__main__':
    main()



