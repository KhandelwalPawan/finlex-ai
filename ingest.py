# importing libraries

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from pathlib import Path

# looping over all the docs

directory = Path('data')
doc_list = list(directory.glob('*.pdf'))
# print(doc_list)

# instantiating splitter

separators = ['\n\n', '\n', ' ', '']
chunk_size = 500
chunk_overlap = 50

splitter = RecursiveCharacterTextSplitter(
    separators= separators,
    chunk_size = chunk_size,
    chunk_overlap = chunk_overlap
)

chunk_list = []

for file in doc_list:
    loader = PyMuPDFLoader(file)
    docs = loader.load()
    chunks = splitter.split_documents(docs)
    chunk_list.extend(chunks)

# embedding documnents

# model_name = "all-MiniLM-L6-v2" 
# embedding_function = HuggingFaceEmbeddings(model_name = model_name)

# vectorstore = FAISS.from_documents(
#     chunk_list,
#     embedding= embedding_function
# )

# vectorstore.save_local('vectorstore')

print(chunk_list[0].metadata)