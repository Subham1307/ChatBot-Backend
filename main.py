import re
import uuid
import os
import io
import json
from typing import List, Optional

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

from utils.uin import extract_uin  # Import UIN extraction function

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load environment variables
load_dotenv()

# Initialize OpenAI LLM and Embeddings
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

# Initialize Qdrant client (local or cloud-hosted)
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")  # Default to local Qdrant
qdrant = QdrantClient(QDRANT_URL)

# Collection name
COLLECTION_NAME = "insurance_policies"

# Ensure Qdrant collection is created
try:
    # Attempt to retrieve the collection. If it does not exist, an error will be raised.
    collection_info = qdrant.get_collection(collection_name=COLLECTION_NAME)
except Exception as e:
    # If the collection is not found, create it.
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
    )

# Global sets/dicts to track UINs and company names
valid_uins = set()
uin_to_company = {}  # Mapping of UIN to company name

# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", "(?<=\. )", " ", ""]
)

# Prompt template for extracting company name
company_prompt_template = """
You are an expert in extracting company names from insurance policy documents.
Extract and return only the company name from the following text.
If no company name is found, return "Not Specified".

Text:
{page_text}
"""

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from entire PDF file."""
    reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
    return text

def extract_text_from_first_two_pages(file_bytes: bytes) -> str:
    """Extract text from the first two pages of a PDF."""
    reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    pages_text = []
    for i in range(min(2, len(reader.pages))):
        page_text = reader.pages[i].extract_text()
        if page_text:
            pages_text.append(page_text)
    return " ".join(pages_text)

@app.post("/upload-pdf")
async def pdf_upload(file: UploadFile = File(...)):
    """Uploads a PDF, extracts its text, and stores it in Qdrant with embeddings.
    Additionally, extracts the company name from the first two pages and maps it to the UIN.
    """
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a PDF.")

    file_bytes = await file.read()
    text = extract_text_from_pdf(file_bytes)

    # Extract UIN from the full document text
    uin = extract_uin(text)
    if not uin:
        raise HTTPException(status_code=400, detail="No valid UIN found in the document.")

    # Check if the UIN already has embeddings in Qdrant
    existing_embeddings = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=[0] * 1536,  # Dummy vector, since we are using a filter only
        query_filter=Filter(must=[FieldCondition(key="uin", match=MatchValue(value=uin))]),
        limit=1  # We only need to check if at least one exists
    )

    if existing_embeddings:
        valid_uins.add(uin)
        first_two_pages_text = extract_text_from_first_two_pages(file_bytes)
        company_prompt = PromptTemplate(input_variables=["page_text"], template=company_prompt_template)
        formatted_company_prompt = company_prompt.format(page_text=first_two_pages_text)
        company_result = llm.invoke(formatted_company_prompt)
        company_name = company_result.content.strip()

        # Map UIN to company name
        uin_to_company[uin] = company_name
        return {"uin": uin, "num_chunks": "chunks are already stored"}

    # Extract company name from the first two pages using LLM
    first_two_pages_text = extract_text_from_first_two_pages(file_bytes)
    company_prompt = PromptTemplate(input_variables=["page_text"], template=company_prompt_template)
    formatted_company_prompt = company_prompt.format(page_text=first_two_pages_text)
    company_result = llm.invoke(formatted_company_prompt)
    company_name = company_result.content.strip()

    # Map UIN to company name
    uin_to_company[uin] = company_name

    # Split the full text into chunks
    chunks = text_splitter.split_text(text)

    # Convert text chunks to embeddings
    vectors = embeddings.embed_documents(chunks)

    # Insert vectors into Qdrant
    points = [
        PointStruct(id=uuid.uuid4().hex, vector=vector, payload={"uin": uin, "text": chunk})
        for vector, chunk in zip(vectors, chunks)
    ]
    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)

    valid_uins.add(uin)

    return {"uin": uin, "company_name": company_name, "num_chunks": len(chunks)}


@app.post("/query")
async def query_endpoint(query: str = Body(...), uins: List[str] = Body(...)):
    """
    Processes a query by decomposing it into three subqueries.
    For each subquery:
      1. Retrieve context for each UIN (grouped by company).
      2. Call the LLM to generate an answer.
    Finally, merge the three subquery answers into one final answer.
    """
    if not uins:
        raise HTTPException(status_code=400, detail="No UINs provided")

    # Validate UINs.
    for uin in uins:
        if uin not in valid_uins:
            raise HTTPException(status_code=404, detail=f"UIN {uin} not found")

    # === STEP 1: Decompose the query into 3 subqueries ===
    decomposition_prompt = f"""
You are an expert in decomposing complex insurance-related queries into three distinct, logically structured subqueries. 
Your goal is to break down the given query into a step-by-step reasoning process that enhances retrieval quality from a vector database.

**Guidelines:**
- Each subquery should be independent yet logically connected to the original query.
- Frame them in a way that helps retrieve the most relevant policy details from insurance documents via similarity search.
- Ensure they progressively refine the understanding of the original question.
- Output format:
    Subquery 1: <subquery 1 text>
    Subquery 2: <subquery 2 text>
    Subquery 3: <subquery 3 text>

Original Query: "{query}"
"""
    decomposition_result = llm.invoke(decomposition_prompt)
    decomposition_text = decomposition_result.content.strip()

    # Use regex to extract the three subqueries.
    subqueries = []
    for i in range(1, 4):
        pattern = rf"Subquery {i}:\s*(.+)"
        match = re.search(pattern, decomposition_text)
        if match:
            subqueries.append(match.group(1).strip())
    if len(subqueries) != 3:
        raise HTTPException(status_code=500, detail="Failed to decompose query into 3 subqueries.")

    print(subqueries)
    # Dictionary to hold the answer for each subquery.
    subquery_answers = {}

    # === STEP 2: Process each subquery separately ===
    for idx, subquery in enumerate(subqueries, start=1):
        # For each UIN, retrieve relevant text chunks and group them by company.
        aggregated_context = {}
        for uin in uins:
            company_name = uin_to_company.get(uin, "Unknown Company")

            # Retrieve top 5 relevant chunks for the current subquery from Qdrant.
            subquery_embedding = embeddings.embed_query(subquery)
            search_results = qdrant.search(
                collection_name=COLLECTION_NAME,
                query_vector=subquery_embedding,
                query_filter=Filter(
                    must=[FieldCondition(key="uin", match=MatchValue(value=uin))]
                ),
                limit=5
            )

            if search_results:
                context = "\n\n".join([hit.payload["text"] for hit in search_results])
            else:
                context = "No relevant excerpts found."

            if company_name in aggregated_context:
                aggregated_context[company_name] += "\n\n" + context
            else:
                aggregated_context[company_name] = context

        companies = list(aggregated_context.keys())
        contexts = [aggregated_context[company] for company in companies]

        # Build a prompt for the current subquery.
        subquery_prompt = f"""
You are an insurance policy analysis expert tasked with answering specific questions using extracted document excerpts.
For each index i, companies[i] is the company name and contexts[i] is the policy document excerpts associated with that company.

**Given Information:**
- **Companies:** {companies}
- **Corresponding Policy Contexts:** {contexts}
- **Subquery:** "{subquery}"

**Instructions:**
1. Carefully analyze the provided policy excerpts for each company.
2. Extract only the most relevant details needed to answer the subquery.
3. Provide a structured and well-reasoned response based solely on the given excerpts.
4. If certain details are missing, explicitly state that the information is not available in the provided context.

**Your Response:**
"""
        subquery_result = llm.invoke(subquery_prompt)
        subquery_answers[f"Subquery {idx}"] = subquery_result.content.strip()

    # === STEP 3: Merge the subquery answers into a final answer ===
    merge_prompt = f"""
You are an expert in insurance policy document analysis. Your task is to synthesize multiple structured answers into a single, coherent response.

**Input:**
- Subquery 1: "{subqueries[0]}" 
  **Answer:** {subquery_answers['Subquery 1']}

- Subquery 2: "{subqueries[1]}" 
  **Answer:** {subquery_answers['Subquery 2']}

- Subquery 3: "{subqueries[2]}" 
  **Answer:** {subquery_answers['Subquery 3']}

**Instructions:**
1. Combine the subquery responses into a single, well-structured answer.
2. Ensure logical coherence and avoid redundancy.
3. Maintain clarity and accuracy, strictly relying on the provided answers.
4. Do not add speculative information or generic explanations.
5. Present the final response concisely without introductory or concluding remarks.

**Final Answer:**
"""
    final_merge_result = llm.invoke(merge_prompt)

    return {
        "final_answer": final_merge_result.content.strip(),
        "subqueries": subqueries,
        "subquery_answers": subquery_answers
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
