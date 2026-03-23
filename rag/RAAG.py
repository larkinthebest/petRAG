from pathlib import Path
import langchain_community
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
from langchain.chat_models import init_chat_model
from langchain.tools import tool
import os, sys
from dotenv import load_dotenv

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
    retrieval = vbstore.as_retriever(search_kwargs = {'k': 4, 'fetch_k': 20}, search_type="mmr")
    
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
    
    chain = {'context':retrieval | format_docs,"question": RunnablePassthrough()} | prompt | llm | StrOutputParser()
    
    chain_with_history = RunnableWithMessageHistory(
    chain,
    session_history,
    input_messages_key="question",
    history_messages_key="chat_history",
    )
    
    return chain_with_history

if __name__ == '__main__':
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    if os.path.exists(INDEX_NAME := 'faiss_index'):
       vstoring = FAISS.load_local('faiss_index',embeddings=embedding_model, allow_dangerous_deserialization=True)
    else:    
        document = load_data(data_path)
        if not document:
            print('no documents')
            sys.exit()
        print(f'uploaded {len(document)} documents')
        split = chunking(document)
        vstoring = embedding(split, embedding_model)
    
    rag_chain = chain_pipe(vstoring)
    
    print('system is ready to work\n')
    while True:
        query = input('user: ')
        if query.lower() == 'exit':
            break
        else:
            response = rag_chain.invoke({"question": query},
                                        config={"configurable": {"session_id": "user_1"}})
            print(f'\nai: {response}\n')