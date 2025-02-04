import re
import uuid
import os
import io
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
qdrant.recreate_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE)  # OpenAI embedding size
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

# Prompt template for generating answers
prompt_template = """
You are an insurance policy document analysis expert specializing in extracting, analyzing, and presenting precise insights from insurance policy documents. Your task is to generate an accurate response using only the provided policy content while ensuring clarity, completeness, and structured formatting.

### Guidelines for Response:
#### 1. Accurate Extraction:
- Extract only relevant details from the policy document.
- Ensure responses are precise, complete, and directly based on the context provided.
- Do NOT provide financial, investment, or legal advice.

#### 2. Structured & Professional Formatting:
  - Clear, concise, and well-formatted answers:
  - Use bullet points or numbered lists where applicable.
  - Present information in an organized and professional manner.

- Use a structured breakdown of key financial figures, highlighting limits, charges, and benefits.

---

### Policy Document Context:**
{context}  

### User's Query:
{query}  

### Answer:

"""

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
    """Processes a query by searching relevant PDF content and generating a response.
    The final answer will include the company name followed by the answer for each UIN.
    """
    if not uins:
        raise HTTPException(status_code=400, detail="No UINs provided")

    # Validate UINs
    for uin in uins:
        if uin not in valid_uins:
            raise HTTPException(status_code=404, detail=f"UIN {uin} not found")

    # Convert query to embedding
    query_embedding = embeddings.embed_query(query)

    answers_by_uin = {}

    for uin in uins:
        # Filter by UIN before performing similarity search
        search_results = qdrant.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding,
            query_filter=Filter(
                must=[FieldCondition(key="uin", match=MatchValue(value=uin))]
            ),
            limit=5  # Retrieve top 5 results
        )

        if not search_results:
            answers_by_uin[uin] = "No relevant information found in the provided document."
        else:
            # Build context from top retrieved chunks
            context = "\n\n".join([hit.payload["text"] for hit in search_results])

            # Create prompt for generating answer
            prompt = PromptTemplate(input_variables=["context", "query"], template=prompt_template)
            formatted_prompt = prompt.format(context=context, query=query)

            # Get LLM response
            answer = llm.invoke(formatted_prompt)
            answers_by_uin[uin] = answer.content

    # Merge answers into a final response in the format: "Company Name: answer"
    final_answer_parts = []
    for uin in uins:
        company_name = uin_to_company.get(uin, "Unknown Company")
        final_answer_parts.append(f"{company_name}: {answers_by_uin[uin]}")
    final_answer = "\n\n".join(final_answer_parts)

    return {
        "final_answer": final_answer,
        "individual_answers": answers_by_uin
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
