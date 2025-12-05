# will mostly use notes and code from classes: 

# step1:
# # Read the original PDFs
# Use pdfplumber

# import os
# import pdfplumber


# pdf_texts = {}

# for file in os.listdir("my_files"):
#     if file.endswith(".pdf"):
#         with pdfplumber.open(os.path.join("my_files", file)) as pdf:
#             # Join all pages into one string
#             pdf_texts[file] = "\n".join(page.extract_text() or "" for page in pdf.pages)

# # Print the full text of each PDF
# for name, text in pdf_texts.items():
#     print(f"\n--- {name} ---\n")
#     print(text)  # Prints the entire text



# use llama_index
# Import PDFReader from LlamaIndex
from llama_index.readers.file import PDFReader 
import os

read = PDFReader()

# load the PDFs
pdf1_reads = read.load_data("my_files/file1.pdf")
pdf2_reads = read.load_data("my_files/file2.pdf")

# print(pdf1_reads[0].text) # ==> means printing only the first pagre
# print(pdf2_reads[0].text) # ==> means printing only the first pagre


# so print out the full pdf, we must loop through all documents/pdf
# full pdf1:
for i, pdf in enumerate(pdf1_reads):
    print(f"\n--- Page/Chunk {i+1} ---\n")
    print(pdf.text)

# full pdf2
for i, pdf in enumerate(pdf2_reads):
    print(f"\n--- Page/Chunk {i+1} ---\n")
    print(pdf.text)

