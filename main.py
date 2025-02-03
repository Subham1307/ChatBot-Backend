from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
import uuid
import os
import io
import PyPDF2
from dotenv import load_dotenv
from typing import List
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
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

prompt_template = """
You are an AI assistant that strictly answers questions based on the provided insurance policy document.
Follow these steps to generate an accurate response:

1. Understand the query and examine what it wants.
2. Analyze the context and get the answer
3. Extract the company name from the insurance policy document.
4. Extract only the necessary information from the insurance policy.
5. Do NOT provide financial, investment, or legal advice.

### Context from PDF:  
{context}  

### User's Latest Question:  
{query}  

Answer:
- First, determine the insurance company name.
- If the company name is found (e.g., "SBI"), structure the response as:
  **SBI: [Extracted Information]**
- Otherwise, provide the extracted answer directly.

Answer:
"""

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from PDF file."""
    reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
    return text

@app.post("/upload-pdf")
async def pdf_upload(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a PDF.")

    file_bytes = await file.read()
    text = extract_text_from_pdf(file_bytes)
    
    # Split text into chunks
    chunks = text_splitter.split_text(text)

    # Generate unique UIN
    uin = str(uuid.uuid4())

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
