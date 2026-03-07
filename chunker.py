from langchain_chroma import Chroma
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

text_embedder = HuggingFaceEmbeddings(model_name='intfloat/multilingual-e5-large')
text_splitter = SemanticChunker(text_embedder, breakpoint_threshold_type="percentile", breakpoint_threshold_amount=97)


def ingest_book(path:str, db: Chroma):
    with open(path, 'r', encoding='utf-8') as file:
        text = file.read()
    chunks = text_splitter.create_documents([text])
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        print("Processing chunk {} of {}".format(i, len(chunks)))
        db.add_documents(chunks[i:i+batch_size])


def get_db(path:str) -> Chroma:
    return Chroma(persist_directory=path, embedding_function=text_embedder, collection_name="books")



