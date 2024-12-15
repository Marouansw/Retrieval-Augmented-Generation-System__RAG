import json
from typing import Annotated,Tuple
from langchain_huggingface import HuggingFaceEmbeddings
from zenml import step
from langchain_community.vectorstores import FAISS
from zenml.types import MarkdownString
from zenml.types import JSONString


@step
def retrieve_documents(query: str, vector_store_path: str) -> Tuple[Annotated[MarkdownString,"Context"],Annotated[JSONString,"Relevent_Chunks"],]:
    
    """Retrieve relevant documents using FAISS."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vector_store = FAISS.load_local(vector_store_path,embeddings,allow_dangerous_deserialization=True)
    
    # docs = vector_store.similarity_search(query, k=3)  # Retrieve top 3 results
    retriever = vector_store.as_retriever()
    retrieved_docs = retriever.invoke(query)
    retrieved_docs = [doc.page_content for doc in retrieved_docs]
    
    chunks_dict = {}
    for i,v in enumerate(retrieved_docs):
        chunks_dict[i]=v
        
    # json_chunks = json.dumps(chunks_dict)

    
    context = "\n".join(retrieved_docs)
    
    return MarkdownString(context),JSONString(chunks_dict)
