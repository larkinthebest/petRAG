from pathlib import Path
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from pypdf import PdfReader
from docx import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_classic.retrievers import ContextualCompressionRetriever
import os, sys
from dotenv import load_dotenv
import streamlit as st

load_dotenv()
model_name = os.getenv('LLM_model', 'gemini-2.5-flash-lite')
data_path = Path(os.getenv('DATA_DIR', './data'))
if not data_path.exists():
    raise TypeError('wrong data path')

def load_data(docs_dir):
    docs = []
    for data in docs_dir.iterdir():
        suffix = data.suffix.lower()
        if suffix == '.txt':
            text = data.read_text(encoding='utf-8')
            if text:
                docs.append(
                {"text": text,
                 'source': data.name,
                 "page": 1
                 })
        elif suffix == '.pdf':
            try:
                reader = PdfReader(data)
            except Exception as e:
                print(f"Error reading {data.name}: {e}")
                continue
            for page_number,page in enumerate(reader.pages, start = 1):
                text = page.extract_text()
                if text:
                    docs.append({"text": text,
                    'source': data.name,
                    "page": page_number})
        elif suffix == '.docx':
            doc = Document(data)
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            text = '\n'.join(paragraphs)
            if text:
                docs.append({"text": text,
                 'source': data.name,
                 "page": 1})
    return docs
             
             
def chunking(document : list[dict]):
    splitted_docs = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 150)
    for doc in document:
        text = doc['text']
        source = doc['source']
        page = doc['page']
        chunks = text_splitter.split_text(text)
        for i, chunk in enumerate(chunks):
            splitted_docs.append({'text': chunk,'source':source,'page': page, 'chunk_id': i})
        print(f'splitted into {len(chunks)} chunks')
    return splitted_docs

def embedding(chunks: list[dict], embedding_model):
    datas = [chunk['text'] for chunk in chunks]
    metadatas = [{'source': chunk['source'], 'page': chunk['page'], 'chunk_id': chunk['chunk_id']} for chunk in chunks] 
    vectorstore = FAISS.from_texts(
        texts=datas,
        embedding=embedding_model,
        metadatas=metadatas)
    vectorstore.save_local('faiss_index')
    return vectorstore


def chain_pipe(vbstore):
    retrieval = vbstore.as_retriever(search_kwargs = {'k': 20})
    reranker = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    kompresor = CrossEncoderReranker(model=reranker, top_n=3)
    compression_retriever = ContextualCompressionRetriever(base_compressor=kompresor, base_retriever=retrieval)
    
    template = """You are a helpful assistant that must take information only from the provided context, not the internet.
    Chat History: {chat_history}
    Context: {context}
    Question: {question}
    always indicate the source of the information in the format [source, page] after each piece of information you use to answer the question. If there are multiple sources, indicate them all.
    If you didn't find appropriate information in the context, you have to write that you didn't find anything related to the question. Do not make up an answer.
    """
    llm = ChatGoogleGenerativeAI(model=model_name, temperature=0)
    prompt = ChatPromptTemplate.from_template(template)
    chat_history_store = ChatMessageHistory()
    def session_history(session_id: str):
        return chat_history_store 
    def format_docs(docs):
        return "\n".join([f'text: {doc.page_content}\n\nsource: {doc.metadata["source"]}, page: {doc.metadata["page"]}' for doc in docs])
    chain = {
        'context': (lambda x: x['question']) | compression_retriever | format_docs,
        'question': lambda x: x['question']
    } | prompt | llm | StrOutputParser()
        
    chain_with_history = RunnableWithMessageHistory(
    chain,
    session_history,
    input_messages_key="question",
    history_messages_key="chat_history",
    )
    
    return chain_with_history


@st.cache_resource
def init_system():
    
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    
    
    if os.path.exists('faiss_index'):
        vstoring = FAISS.load_local('faiss_index', embeddings=embedding_model, allow_dangerous_deserialization=True)
    else:    
        document = load_data(data_path)
        if not document:
            st.error('No docs in folder data, please add some documents and reload the page')
            sys.exit()
        split = chunking(document)
        vstoring = embedding(split, embedding_model)
    
    
    rag_chain = chain_pipe(vstoring)
    return rag_chain


st.title("🤖 my RAG")


rag_chain = init_system()

if "messages" not in st.session_state:
    st.session_state.messages = []


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


if prompt := st.chat_input("Ask a question about your documents..."):

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)


    with st.chat_message("assistant"):
       
        with st.spinner("Analyzing documents..."):
            
            response = rag_chain.invoke({"question": prompt},
                config={"configurable": {"session_id": "user_1"}})
            
            
            st.markdown(response)
            

    st.session_state.messages.append({"role": "assistant", "content": response})