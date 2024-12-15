from steps.load import import_data
from steps.split import split_data
from steps.vectore_store import embeddingsXvectore_store
from steps.relevent_docs import retrieve_documents
from steps.generate_response import generate_response
from zenml import pipeline
from zenml.types import MarkdownString


@pipeline
def rag_pipeline(query : str):
    raw_data = import_data()  # Run the import_data step
    splitted_data = split_data(raw_data)
    store_data_dir = embeddingsXvectore_store(splitted_data)
    context,relevent_chunks_to_query = retrieve_documents(query,store_data_dir)
    response = generate_response(context,query)