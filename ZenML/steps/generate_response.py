from zenml import step
from typing import Annotated, List, Tuple
from langchain_ollama.llms import OllamaLLM
from langchain_google_genai import GoogleGenerativeAI
from zenml.types import MarkdownString

@step
def generate_response(context:MarkdownString,query:str) ->Tuple[Annotated[MarkdownString,"LLAMA's 3.2 Response"],Annotated[MarkdownString,"Gemini-1.5-pro's Response"]] :
    """Generate a response using a language model."""
    # LLAMA
    llm=OllamaLLM(model='llama3.2:latest')
    # GEMINI
    api_key = 'AIzaSyDPY7XqjkX_G-hiS-xaQ0oMuEuAIDXNywU'
    llm2 = GoogleGenerativeAI(model="models/gemini-1.5-pro-latest",google_api_key=api_key)
    # PROMPT
    # context = "\n".join(retrieved_docs)
    prompt = f"Reponds a la question suivante en se basant sur le context :\n\nContext:\n{context}\n\nQuestion:\n{query}"
    # RESPONSES
    response1 = llm.invoke(prompt)
    response2 = llm2.invoke(prompt)
    return MarkdownString(str(response1)),MarkdownString(str(response2))
