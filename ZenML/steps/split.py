from langchain_text_splitters import CharacterTextSplitter
from zenml import step
from typing import Annotated


@step
def split_data(pdf_data:list) -> Annotated[list,"Chunks"]: 
    text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=80,
)
    chunks=text_splitter.split_documents(pdf_data)
    return chunks