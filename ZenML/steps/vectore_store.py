from pathlib import Path
from zenml import step
from typing import Annotated

# from zenml.utils import Output

import langchain_core
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


@step
def embeddingsXvectore_store(doc:list) -> Annotated[str,"embeddings_file"]:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vector_store = FAISS.from_documents(documents=doc, embedding=embeddings)
    # retriever = vector_store.as_retriever()
    # faiss_file = (r"C:/Users/LENOVO/AppData/Roaming/zenml/local_stores/faiss_index")
    faiss_file = ("./faiss_index")

    vector_store.save_local(faiss_file)
    return faiss_file
