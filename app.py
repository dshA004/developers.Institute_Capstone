# venv\Scripts\activate
import streamlit as st
from pypdf import PdfReader
from dotenv import load_dotenv
from langchain_text_splitters import CharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
# from transformers import pipeline
# import torch
from groq import Groq
from langchain_groq import ChatGroq
import time 


load_dotenv()
import os
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
# ----------------------------------------------------------------------------------- #

st.set_page_config(
        page_title="Mini PDF AI Assistant",
        page_icon=":books:", 
        layout="wide"
        )

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

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
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
# device = "cpu"  # force CPU
# embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

embedding_model = load_embedding_model()
client = Groq()
GROQ_MODEL = "llama-3.3-70b-versatile"


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

# ----------------------------------------------------------------------------------------------------------------- #

# st.text_input("Ask a questions about the document!")

with st.sidebar:
    st.sidebar.title("Zola!") #  AI Assistant
    st.sidebar.title("Mini Study Buddy!") #  AI Assistant
    st.sidebar.markdown("Summarize, Chat and Generate questions.")
    st.header("Upload PDFs")

    uploaded_files = st.file_uploader(
                            "Upload PDF",
                           type = "pdf",
                            accept_multiple_files=True
                            )

    # if st.button("Process"):
    #       with st.spinner("Proccessing"):
process_button = st.sidebar.button("Process PDFs")
# ------------------------------------------------------------------------------- #

# Main Page
st.header("AI Assistant:books:")
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

            # if extracted_text:
            #     st.toast("Text extracted successfully")
            # MAKE THE NOTIFICATION DISSAPEAR FASTER!
            # if extracted_text:
            #     msg1 = st.success("Text extracted successfully! ✅")
            #     time.sleep(1)
            #     msg1.empty()
   


            # else:
            #     st.warning("No text found in the uploaded PDFs.")


            # if st.session_state.chunks:
            #     with st.spinner("Creating embeddings and FAISS vector database..."):
            #         st.session_state.faiss_index, st.session_state.embeddings = create_faiss_index(st.session_state.chunks)
            # # st.toast("Embeddings and FAISS index ready")
            # msg2 = st.success("Embeddings and FAISS index ready! ✅")
            # time.sleep(1)
            # msg2.empty()
        # else:
        #     st.warning("Upload a PDF")


            if extracted_text:
                if st.session_state.chunks:
                    with st.spinner("Please wait..."):
                        st.session_state.faiss_index, st.session_state.embeddings = create_faiss_index(st.session_state.chunks)
                msg1 = st.success("Text extracted successfully! ✅")
                time.sleep(1.0)
                msg1.empty()
            else:
                st.warning("No text found in the uploaded PDFs.")


st.divider()
# --------------------------------------------------------------------------------------------------------------------------------------------------- #
# DEBUG CONFIRMATION (TEMP)
# =========================
# if st.session_state.pdf_text:
#         st.write("✅ Text extracted")

# if st.session_state.chunks:
#         st.write("✅ Chunks ready")

# if st.session_state.embeddings is not None:
#     st.write("✅ Created embeddings")

# st.divider()
             
# SUMMARISATION: --------------------------------------------------------------------------------------------------------------------------
st.subheader("📝 Summarisation")

summarise_button = st.button("Generate Summary")

       

if summarise_button:
    if st.session_state.chunks:
        with st.spinner("Summary is being generated..."):
            # Combine all chunks into one text
            combined_text = "\n".join(st.session_state.chunks)
            combined_text = combined_text[:8000]
            if len("\n".join(st.session_state.chunks)) > 8000:
               msg = st.warning("⚠️ Your PDF is too large, only the first part will be processed.")
               time.sleep(2)
               msg.empty()
            
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
            # summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)  # -1 for CPU
            # summary_text = summarizer(combined_text, max_length=250, min_length=80, do_sample=False)[0]['summary_text']
            # # Store summary in session state
            # # st.session_state.summary = summary
            # st.session_state.summary = summary_text

            
            # st.success("Summary generated ✅")
            # # st.write(summary)
            # st.write(summary_text)

            # st.info("Summarization coming soon via Groq!")
        try:
            response = client.chat.completions.create(
                # model="llama-3.3-70b-versatile",
                model=GROQ_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are Zola, a helpful assistant that summarizes documents clearly and concisely."
                    },
                    {
                        "role": "user",
                        "content": f"Please summarize the following document:\n\n{combined_text}"
                    }
                ]
            )
            summary_text = response.choices[0].message.content
            st.session_state.summary = summary_text
            msg = st.success("Summary generated ✅")
            time.sleep(1.5)
            msg.empty()
            st.write(summary_text)
        except Exception as e:
             st.error("⚠️ Zola is unavailable right now. Please try again!")

    else:
        st.warning("Please upload a PDF.")

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

# if st.button("Clear Chat 🗑️"):
#      st.session_state.chat_history = []
#      st.rerun()

for message in st.session_state.chat_history:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.write(message["content"])
    else:
        with st.chat_message("assistant"):
            st.write(message["content"])

        
user_question = st.text_input("Ask a question about the PDFs:",
                              key="active_question_input")
ask_button = st.button("Ask", key="ask_button")

if st.button("Clear Chat 🗑️"):
     st.session_state.chat_history = []
     st.rerun()
              
if ask_button:
    if not user_question.strip():
         st.warning("⚠️ Please type a question first!")
    elif st.session_state.chunks and st.session_state.embeddings is not None:
        with st.spinner("Searching for answer..."):
            results = search(user_question)
            context = "\n".join(results)
            context = context[:8000]
            if len("\n".join(st.session_state.chunks)) > 8000:
                msg = st.warning("⚠️ Your PDF is too large, only the first part will be processed.")
                time.sleep(2)
                msg.empty()


            # Build messages with chat history
            messages = [
                {
                    "role": "system",
                    "content": "You are Zola, a helpful assistant that answers questions based on the provided document context."
                }
            ]

            # Add chat history
            for msg in st.session_state.chat_history:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })

            # Add current question
            messages.append({
                "role": "user",
                "content": f"Based on the following context:\n\n{context}\n\nAnswer this question: {user_question}"
            })


        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages
            )
            answer = response.choices[0].message.content
            
            # Save to chat history
            st.session_state.chat_history.append({"role": "user", "content": user_question})
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            st.rerun()
        except Exception as e:
             st.error("⚠️ Zola is unavailable right now. Please try again!")
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

st.subheader("❓ Questions ")
num_questions = st.slider("How many questions?", min_value=1, max_value=10, value=5)
generate_button = st.button("Generate questions")
if generate_button:
     if not st.session_state.chunks:
                    st.warning("Please do upload a PDF and Process again!")
     else:
          with st.spinner("Questions are being generated..."):
            #    st.info("Questions coming soon!")


            combined_text = "\n".join(st.session_state.chunks)
            combined_text = combined_text[:8000]
            if len("\n".join(st.session_state.chunks)) > 8000:
               msg = st.warning("⚠️ Your PDF is too large, only the first part will be processed.")
               time.sleep(2)
               msg.empty()

            
            try:
                response = client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are Zola, a helpful assistant that generates insightful questions based on a document."
                        },
                        {
                            "role": "user",
                            "content": f"Based on the following document, generate {num_questions} thoughtful questions that test understanding of the key concepts. For each question, provide the answer too. Format your response exactly like this:\n\nQ1: question here\nA1: answer here\n\nQ2: question here\nA2: answer here\n\nDocument:\n\n{combined_text}"
                        }
                    ]
                )
                questions = response.choices[0].message.content
                msg = st.success("Questions generated ✅")
                time.sleep(1.5)
                msg.empty()
                lines = questions.strip().split("\n\n")
                for pair in lines:
                    lines_in_pair = pair.strip().split("\n")
                    if len(lines_in_pair) >= 2:
                        question = lines_in_pair[0]
                        answer = lines_in_pair[1]
                        st.write(f"**{question}**")
                        with st.expander("Answer "):
                            st.write(answer)
                        st.divider()
            except Exception as e:
                st.error("⚠️ Zola is unavailable right now. Please try again!")