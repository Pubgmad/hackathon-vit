import cassio
from dotenv import load_dotenv
import os
from langchain_community.vectorstores import Cassandra
from langchain_huggingface import HuggingFaceEmbeddings
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from typing import Literal
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_community.document_loaders import WebBaseLoader
from langchain_groq import ChatGroq
from langchain.schema import Document
from typing import List
from langgraph.graph import END, StateGraph, START
from langchain.tools import Tool
from typing_extensions import TypedDict
from langchain_community.tools import SerpAPIWrapper  

load_dotenv()

USER_AGENT = os.environ.get("USER_AGENT")
ASTRA_DB_APPLICATION_TOKEN = os.environ.get("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_ID = os.environ.get("ASTRA_DB_ID")
SERP_API_KEY = os.getenv("SERPAPI_API_KEY")  

cassio.init(token=ASTRA_DB_APPLICATION_TOKEN, database_id=ASTRA_DB_ID)

urls = [
    "https://rmi.org/the-ev-battery-supply-chain-explained/",
    "https://www.iea.org/reports/ev-battery-supply-chain-sustainability",
    "https://www.sciencedirect.com/science/article/abs/pii/S0921344921004882",
]
docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]



text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=500, chunk_overlap=200)
doc_splits = text_splitter.split_documents(docs_list)


embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


astra_vector_store = Cassandra(
    embedding=embeddings,
    table_name="qa_mini_demo",
    session=None,
    keyspace=None
)
astra_vector_store.add_documents(doc_splits)
print(f"Inserted {len(doc_splits)} documents.")

astra_vector_index = VectorStoreIndexWrapper(vectorstore=astra_vector_store)
retriever = astra_vector_store.as_retriever()


class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""
    datasource: Literal["vectorstore", "wiki_search", "web_search"] = Field(
        ..., description="Choose vectorstore, Wikipedia, or web search."
    )


groq_api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Gemma2-9b-It")
structured_llm_router = llm.with_structured_output(RouteQuery)
print("ChatGroq initialized successfully with structured output!")

system = """You are an expert in routing user queries which are only relevant to supply chain mainly focused on batteries, chips. 
Choose from:
1. 'vectorstore' for queries on battery supply chains.
2. 'wiki_search' for general knowledge.
3. 'web_search' if neither source is relevant.
"""


route_prompt = ChatPromptTemplate.from_messages([
    ("system", system),
    ("human", "{question}"),
])
question_router = route_prompt | structured_llm_router

arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)
api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper)


web_search_tool = SerpAPIWrapper(serpapi_api_key=SERP_API_KEY)


class GraphState(TypedDict):
    """Represents the state of our graph."""
    question: str
    generation: str
    documents: List[Document]


def retrieve(state):
    """Retrieve documents from the vector store."""
    print("---RETRIEVE---")
    question = state["question"]
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}

def wiki_search(state):
    """Search Wikipedia."""
    print("---WIKIPEDIA SEARCH---")
    question = state["question"]
    wiki_results = wiki.invoke({"query": question})
    wiki_results = [Document(page_content=wiki_results)]
    return {"documents": wiki_results, "question": question}

def web_search(state):
    """Search the web in real-time using SerpAPI."""
    print("---WEB SEARCH---")
    question = state["question"]
    search_results = web_search_tool.run(question)
    web_results = [Document(page_content=search_results)]
    return {"documents": web_results, "question": question}

def route_question(state):
    """Route to Wikipedia, Vectorstore, or Web Search."""
    print("---ROUTE QUESTION---")
    question = state["question"]
    source = question_router.invoke({"question": question})
    
    if source["datasource"] == "wiki_search":
        print("---ROUTING TO WIKIPEDIA SEARCH---")
        return "wiki_search"
    elif source["datasource"] == "vectorstore":
        print("---ROUTING TO VECTORSTORE---")
        return "retrieve"
    elif source["datasource"] == "web_search":
        print("---ROUTING TO WEB SEARCH---")
        return "web_search"


workflow = StateGraph(GraphState)

workflow.add_node("wiki_search", wiki_search)
workflow.add_node("retrieve", retrieve)
workflow.add_node("web_search", web_search)

workflow.add_conditional_edges(
    START,
    route_question,
    {
        "wiki_search": "wiki_search",
        "vectorstore": "retrieve",
        "web_search": "web_search",
    },
)

workflow.add_edge("retrieve", END)
workflow.add_edge("wiki_search", END)
workflow.add_edge("web_search", END)

app = workflow.compile()


try:
    from IPython.display import Image, display
    display(Image(app.get_graph().draw_mermaid_png()))
except Exception:
    pass
