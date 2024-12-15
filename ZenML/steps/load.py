from langchain_community.document_loaders import PyPDFLoader
from zenml import step
from typing import Annotated


@step
def import_data() -> Annotated[list,"My_Data"] : 
# Replace this with your actual document or whatever (Web page, Word...)
    loader = PyPDFLoader(r"C:\Users\LENOVO\Desktop\DATA_SCIENCE\Deep_Learning\ReGenuis--RAG\loi-n-01-00-portant-organisation-de-lenseignement-supérieur.pdf")
    data = loader.load()
    return data