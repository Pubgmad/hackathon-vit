from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware  
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv


load_dotenv()


app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

@app.get("/")
def read_root():
    return {"message": "Hello, world!"}


groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY is not set in the environment!")


llm = ChatGroq(groq_api_key=groq_api_key, model_name="Gemma2-9b-It")

route_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert AI assistant."),
    ("human", "{question}"),
])


class QueryRequest(BaseModel):
    question: str


@app.post("/chat/")
async def chat(request: QueryRequest):
   
    response_chain = route_prompt | llm 
    answer = response_chain.invoke({"question": request.question}) 
    return {"response": answer}  
