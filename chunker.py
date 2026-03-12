import os
import re
import gc
import torch
from typing import Dict
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

text_embedder = HuggingFaceEmbeddings(model_name='intfloat/multilingual-e5-large')
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2500,
    chunk_overlap=300,
    separators=["\n\n", "\n", ".", " ", ""]
)


def get_metadata_from_filename(path: str) -> Dict[str, str]:
    """
    Извлекает метаданные из формата 'Имя Автора-Название Книги.txt'
    """
    basename = os.path.basename(path).replace('.txt', '')

    if "-" in basename:
        parts = basename.split("-", 1)
        author = parts[0].strip().title()
        title = parts[1].strip().title()
    else:
        author = "Неизвестен"
        title = basename.replace('-', ' ').strip().title()

    return {"author": author, "title": title}


def ingest_book(path: str, db: Chroma):
    meta = get_metadata_from_filename(path)
    if not os.path.exists(path):
        raise FileExistsError("File not exists")
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        full_text = f.read()
    chapter_pattern = r'(?i)^\s*(Глава\s+(?:[IVXLCDM]+|[0-9]+|первая|вторая|третья|четвертая|пятая|шестая|седьмая|восьмая|девятая|десятая))\s*$'
    parts = re.split(chapter_pattern, full_text, flags=re.MULTILINE)
    final_docs = []
    if parts[0].strip():
        chunks = text_splitter.split_documents([Document(page_content=parts[0])])
        for c in chunks:
            c.metadata.update({"author": meta["author"], "book": meta["title"], "chapter": "Вступление"})
            final_docs.append(c)
    for i in range(1, len(parts), 2):
        chapter_title = parts[i].strip().title()
        chapter_content = parts[i + 1] if i + 1 < len(parts) else ""
        if chapter_content.strip():
            chunks = text_splitter.split_documents([Document(page_content=chapter_content)])
            for c in chunks:
                c.metadata.update({
                    "author": meta["author"],
                    "book": meta["title"],
                    "chapter": chapter_title
                })
                final_docs.append(c)
    db.add_documents(final_docs)
    clear_hardware_cache()


def clear_hardware_cache():
    """
    Очищает оперативную память и видеопамять.
    """
    gc.collect()
    if torch.cuda.is_available():
        with torch.cuda.device('cuda'):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        print("[+] VRAM очищена.")
    gc.collect()
    print("[+] RAM очищена.")


def get_db(path: str) -> Chroma:
    return Chroma(persist_directory=path, embedding_function=text_embedder, collection_name="library")
