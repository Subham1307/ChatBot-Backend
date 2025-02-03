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

valid_uins = set()

# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", "(?<=\. )", " ", ""]
)

# Prompt template for generating answers
prompt_template = """
You are an AI assistant specializing in extracting, analyzing, and presenting precise insights from insurance policy documents. Your task is to generate an accurate response using only the provided policy content while ensuring clarity, completeness, and structured formatting.

### **Guidelines for Response:**
#### 1. **Accurate Extraction**:
- Extract only relevant details from the policy document.
- Ensure responses are **precise, complete, and directly based on the context** provided.
- **Do NOT provide financial, investment, or legal advice.**

#### 2. **Structured & Professional Formatting**:
- **Company Identification**: If the insurance company's name is mentioned, start the response with its name. Example:  
  **SBI: [Extracted Information]**
- **Clear, concise, and well-formatted answers**:
  - Use bullet points or numbered lists where applicable.
  - Present information in an **organized and professional** manner.

#### 3. **Comprehensive Numerical Extraction**:
- **Collect as many numerical details as possible**, including percentages, monetary values, age limits, durations, charges, and benefits.
- Ensure all numerical details follow a standardized format:
  - **Monetary values**: Use INR format with commas (e.g., **₹50,000**).
  - **Percentages**: Express values explicitly (e.g., **"1.8% loyalty additions"**).
  - **Age and duration**: Use whole numbers (e.g., **"30 years"**).
- If a charge, benefit, or limit applies over time, **aggregate values** where relevant:
  - **Example**: "Policy Admin Charge: ₹200/month, totaling ₹2,400 annually."
  - **Example**: "Maximum of 4 partial withdrawals per year, with a ₹100 charge per withdrawal beyond the free limit."

#### 4. **Handling Missing or Ambiguous Data**:
- If a required detail is **not explicitly mentioned**, respond with `"Not Specified"`.
- Avoid making assumptions but provide logical inferences if evident from the context.
- Clearly indicate when inferred details are used.

#### 5. **Scenario-Based Summaries (if applicable)**:
- Where relevant, add structured summaries or examples, such as:
  - **"Premium payment term: 10 years, policy maturity at 20 years."**
  - **"Sum assured: ₹5,00,000 with a 10% bonus every 5 years."**
  - **"Partial withdrawal allowed after 5 years, subject to a 5% fee per transaction."**
- Use a structured breakdown of key financial figures, highlighting limits, charges, and benefits.

---

### **Policy Document Context:**
{context}  

### **User's Query:**
{query}  

### **Answer:**

"""

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from PDF file."""
    reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
    return text

@app.post("/upload-pdf")
async def pdf_upload(file: UploadFile = File(...)):
    """Uploads a PDF, extracts its text, and stores it in Qdrant with embeddings."""
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a PDF.")

    file_bytes = await file.read()
    text = extract_text_from_pdf(file_bytes)

    # Extract UIN from document
    uin = extract_uin(text)
    if not uin:
        raise HTTPException(status_code=400, detail="No valid UIN found in the document.")

    # Split text into chunks
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

    return {"uin": uin, "num_chunks": len(chunks)}

@app.post("/query")
async def query_endpoint(query: str = Body(...), uins: List[str] = Body(...)):
    """Processes a query by searching relevant PDF content and generating a response."""
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

            # Create prompt
            prompt = PromptTemplate(input_variables=["context", "query"], template=prompt_template)
            formatted_prompt = prompt.format(context=context, query=query)

            # Get LLM response
            answer = llm.invoke(formatted_prompt)
            answers_by_uin[uin] = answer.content

    # Merge answers into a final response
    final_answer = "\n\n".join([answers_by_uin[uin] for uin in uins])

    return {
        "final_answer": final_answer,
        "individual_answers": answers_by_uin
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
