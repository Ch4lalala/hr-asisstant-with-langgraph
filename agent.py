import os
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from operator import add as add_messages
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o",
    api_key=os.getenv("YOUR_API_KEY"),
    base_url=os.getenv("YOUR_BASE_URL"),
    temperature = 0) 

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=os.getenv("YOUR_API_KEY"),
    base_url=os.getenv("YOUR_BASE_URL"),
)

pdf_path = "cv.pdf"
if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"PDF file not found: {pdf_path}")


pdf_loader = PyPDFLoader(pdf_path) # This loads the PDF
try:
    pages = pdf_loader.load()
    print(f"PDF has been loaded and has {len(pages)} pages")
except Exception as e:
    print(f"Error loading PDF: {e}")
    raise