
"""
Vector store and QA chain construction helpers.
"""

from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import CharacterTextSplitter

from config import embedding_model_name, model_name

def build_vectorstore(docs, in_memory_limit: int = None):
    """
    Build a FAISS vector store from Document objects.
    If `in_memory_limit` is given, only the first N docs will be used (useful for quick testing).
    """
    if in_memory_limit:
        docs = docs[:in_memory_limit]
    embedding_model = OpenAIEmbeddings(model=embedding_model_name)
    vectorstore = FAISS.from_documents(docs, embedding_model)
    # 1) 문서를 작은 청크로 나누기
    splitter = CharacterTextSplitter(
        chunk_size=1000,      # 한 청크당 최대 1,000자
        chunk_overlap=200     # 청크 간 200자 중첩
    )
    split_docs = splitter.split_documents(docs)
    # 2) 분할된 문서에 대해 FAISS 벡터스토어 생성
    vectorstore = FAISS.from_documents(split_docs, embedding_model)
    return vectorstore

def build_qa_chain(vectorstore):
    """Create a conversational QA chain on top of the given vector store."""
    llm = ChatOpenAI(model_name=model_name, temperature=0.3)
    retriever = vectorstore.as_retriever(search_type="similarity", k=3)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        verbose=False
    )
    return qa_chain
