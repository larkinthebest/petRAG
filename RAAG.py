from pathlib import Path
from pypdf import PdfReader
from docx import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

data_path = Path(r"C:\Users\kdyms\Desktop\pets\notjjup\data")
if not data_path.exists():
    raise TypeError('wrong data path')

def load_data(docs_dir):
    docs = []
    for data in docs_dir.iterdir():
        suffix = data.suffix.lower()
        if suffix == '.txt':
            text = data.read_text()
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

def similaritySearch(vbstore):
    
    