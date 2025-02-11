from typing import List
import uuid
import os
import io

from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
import PyPDF2
from dotenv import load_dotenv

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from utils.uin import extract_uin  # Import UIN extraction function

# ---------------------------
# TruLens Imports and Setup
# ---------------------------
from trulens.apps.custom import instrument, TruCustomApp
from trulens.core import TruSession, Feedback, Select
from trulens.providers.openai import OpenAI as TruOpenAI
import numpy as np
from trulens.dashboard import run_dashboard

# Create a TruSession for evaluation recording and reset it.
session = TruSession()
session.reset_database()

# Create a TruLens provider instance (using the same LLM engine as our chain)
provider = TruOpenAI(model_engine="gpt-4o-2024-08-06")

# Define a simple feedback function evaluating answer relevance using inputs and outputs.
f_answer_relevance = Feedback(
    provider.relevance_with_cot_reasons,
    name="Answer Relevance"
).on_input().on_output()

# ---------------------------
# FastAPI and Core App Setup
# ---------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load environment variables
load_dotenv()

# Initialize LLM and Embeddings
llm = ChatOpenAI(model_name="gpt-4o-2024-08-06", temperature=0)
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

# Initialize Qdrant client
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
qdrant = QdrantClient(QDRANT_URL)

# Collection name for policies
COLLECTION_NAME = "insurance_policies"
try:
    # Check if the collection exists
    collection_info = qdrant.get_collection(collection_name=COLLECTION_NAME)
except Exception:
    # Create collection if it does not exist
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
    )

# Global in-memory data stores
valid_uins = set()
uin_to_company = {}  # Mapping of UIN to company name

# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50,
    separators=["\n\n", "\n", "(?<=\. )", " ", ""]
)

# Prompt template for extracting company name from the first two pages of a PDF
company_prompt_template = """
You are an expert in extracting company names from insurance policy documents.
Extract and return only the company name from the following text.
If no company name is found, return "Not Specified".

Text:
{page_text}
"""

def extract_text_from_pdf(file_bytes: bytes) -> str:
    reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
    return text

def extract_text_from_first_two_pages(file_bytes: bytes) -> str:
    reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    pages_text = []
    for i in range(min(2, len(reader.pages))):
        page_text = reader.pages[i].extract_text()
        if page_text:
            pages_text.append(page_text)
    return " ".join(pages_text)

@app.post("/upload-pdf")
async def pdf_upload(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a PDF.")
    file_bytes = await file.read()
    text = extract_text_from_pdf(file_bytes)
    
    # Extract UIN from the full document text
    uin = extract_uin(text)
    if not uin:
        raise HTTPException(status_code=400, detail="No valid UIN found in the document.")
    
    # Check if embeddings for this UIN already exist using a dummy vector
    existing_embeddings = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=[0] * 1536,  # Dummy vector to use the filter only
        query_filter=Filter(must=[FieldCondition(key="uin", match=MatchValue(value=uin))]),
        limit=1
    )
    
    first_two_pages_text = extract_text_from_first_two_pages(file_bytes)
    company_prompt = PromptTemplate(input_variables=["page_text"], template=company_prompt_template)
    formatted_company_prompt = company_prompt.format(page_text=first_two_pages_text)
    company_result = llm.invoke(formatted_company_prompt)
    company_name = company_result.content.strip()
    
    uin_to_company[uin] = company_name
    
    if existing_embeddings:
        valid_uins.add(uin)
        return {"uin": uin, "num_chunks": "chunks are already stored"}
    
    # Split full text into chunks and embed each chunk
    chunks = text_splitter.split_text(text)
    vectors = embeddings.embed_documents(chunks)
    points = [
        PointStruct(id=uuid.uuid4().hex, vector=vector, payload={"uin": uin, "text": chunk})
        for vector, chunk in zip(vectors, chunks)
    ]
    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
    valid_uins.add(uin)
    
    return {"uin": uin, "company_name": company_name, "num_chunks": len(chunks)}

#############################################
#           Query Chain and TruLens          #
#############################################

# Helper function to build aggregated context from Qdrant for each UIN.
def build_aggregated_context(uins: List[str], query: str) -> dict:
    aggregated_context = {}
    for uin in uins:
        company_name = uin_to_company.get(uin, "Unknown Company")
        query_embedding = embeddings.embed_query(query)
        search_results = qdrant.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding,
            query_filter=Filter(must=[FieldCondition(key="uin", match=MatchValue(value=uin))]),
            limit=5
        )
        # Debug: Print search results
        for idx, hit in enumerate(search_results):
            print(f"Result {idx + 1} for {company_name}: {hit.payload['text']}")
        context = "\n\n".join([hit.payload["text"] for hit in search_results]) if search_results else "No relevant excerpts found."
        aggregated_context[company_name] = context
    return aggregated_context

# Prompt template for the final aggregated query
final_prompt_template = """
You are an insurance policy analysis expert tasked with answering a query using extracted document excerpts.
For each index i, companies[i] is the company name and contexts[i] is the policy document excerpts associated with that company.

Given Information:
- Companies: {companies}
- Corresponding Policy Contexts: {contexts}
- Query: "{query}"

Instructions:
1. Analyze the Query and Extract Relevant Information:
   - Thoroughly review the query and the provided contexts.
   - Ensure that your analysis is detailed, accurate, concise, user-friendly, and directly addresses the query.
   - If certain details are missing, explicitly state that the information is not available.

2. Think for Each Company Separately:
   - For each company[i], analyze its associated policy excerpts in contexts[i].
   - Extract all relevant details specific to that company's policies.
   - Think of the deatiled answer for that company in your mind
3. Compile a Final Answer:
   - Summarize the key insights from all the company-specific responses.
   - Ensure the final answer is detailed, comprehensive, and clearly states if any critical information is missing.
4. Ensure that response should not provide any finacial advice, if user asks for financial advice then clearly state that you are unable to give any advice
5. If the question if out of your context then you should respond that you dont know the answer
6. Please provide direct answers without preemptive phrases.
Final Answer:
"""

# Create an LLMChain using the final prompt template.
final_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate(input_variables=["companies", "contexts", "query"], template=final_prompt_template)
)

# Instrument the query_chain function using TruLens.
@instrument
def query_chain(query: str, uins: List[str]) -> dict:
    # Validate UINs.
    for uin in uins:
        if uin not in valid_uins:
            raise HTTPException(status_code=404, detail=f"UIN {uin} not found")
    
    # Build aggregated context by retrieving Qdrant results per UIN.
    aggregated_context = build_aggregated_context(uins, query)
    companies = list(aggregated_context.keys())
    
    citations = []
    for company in companies:
        chunks = aggregated_context[company].split("\n\n")
        for chunk in chunks:
            citations.append({"company": company, "text": chunk, "chunk_id": uuid.uuid4().hex})

    contexts = [c["text"] for c in citations]
    
    final_answer = final_chain.run(companies=companies, contexts=contexts, query=query)
    
    return {
        "final_answer": final_answer.strip(),
        "citations": citations
    }

# Wrap the query_chain in a TruCustomApp with the feedback function.
try:
    tru_query_app = TruCustomApp(
        query_chain,
        app_name="QueryChain",
        app_version="1.0",
        feedbacks=[f_answer_relevance]
    )
    print("TruCustomApp initialized successfully.")
except Exception as e:
    print("Error initializing TruCustomApp:", e)

# Expose the query chain via an endpoint.
@app.post("/query")
async def query_endpoint(query: str = Body(...), uins: List[str] = Body(...)):
    # Execute the instrumented query_chain within the TruCustomApp context.
    with tru_query_app as recording:
         result = query_chain(query, uins)
    return result

run_dashboard(session)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
