#importing libraries
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

#loading Faiss

def get_rag_chain():
    model_name = "all-MiniLM-L6-v2" 
    embedding_function = HuggingFaceEmbeddings(model_name = model_name)

    vectorstore = FAISS.load_local(
        folder_path= 'vectorstore',
        embeddings= embedding_function,
        allow_dangerous_deserialization= True) 

    retriever = vectorstore.as_retriever() 


# instantiating LLM

    llm = ChatOllama(model= 'llama3.2', temperature= 0.2)


    message = """ You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the 
    question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
    \nQuestion: {question} \nContext: {context} \nAnswer: """

    prompt = ChatPromptTemplate.from_template(message)

    retrieval = RunnableParallel(
        context = retriever,  # path 1: documents for the prompt
        question = RunnablePassthrough()  # path 2: original question passes through
    )

    rag_chain = retrieval | RunnableParallel(
        answer = prompt | llm | StrOutputParser(),
        sources = lambda x: list(set([doc.metadata['source'] for doc in x['context']]))
    )
    
    return rag_chain


