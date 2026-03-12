from pathlib import Path
from pypdf import PdfReader
from docx import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
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
data_path = Path(r"C:\Users\kdyms\Desktop\pets\notjjup\data")
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
            reader = PdfReader(data)
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

def embedding(chunks: list[dict]):
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    datas = [chunk['text'] for chunk in chunks]
    metadatas = [{'source': chunk['source'], 'page': chunk['page'], 'chunk_id': chunk['chunk_id']} for chunk in chunks] 
    vectorstore = FAISS.from_texts(
        texts=datas,
        embedding=embedding_model,
        metadatas=metadatas)
    return vectorstore


def chain_pipe(vbstore):
    retrieval = vbstore.as_retriever(search_kwargs = {'k': 3})
    
    template = """You are a helpful assistant that must take information only from the provided context, not the internet.
    Context: {context}
    Question: {question}
    If you didn't find appropriate information in the context, you have to write that you didn't find anything related to the question. Do not make up an answer.
    """
    llm = ChatGoogleGenerativeAI(model=model_name, temperature=0)
    prompt = ChatPromptTemplate.from_template(template)
    def format_docs(docs):
        result = [f'{doc.page_content}' for doc in docs]
        
    chain = {'context':retrieval | format_docs,"question": RunnablePassthrough} | llm | prompt | StrOutputParser()
    
    return chain

if __name__ == '__main__':
    document = load_data(data_path)
    if not document:
        print('no documents')
        sys.exit()
    print(f'uploaded {len(document)} documents')
    
    split = chunking(document)
    vstoring = embedding(split)
    rag_chain = chain_pipe(vstoring)
    
    print('system is ready to work\n')
    while True:
        query = input('user: \n')
        if query.lower() == 'exit':
            break
        else:
            response = rag_chain.invoke(query)
            print(f'ai: {response}\n')