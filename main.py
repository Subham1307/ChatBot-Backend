from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uuid
import os
from dotenv import load_dotenv
import io
import PyPDF2
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from fastapi import Body
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser




app = FastAPI()

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
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

# Initialize FAISS vectorstore
vectorstore = None
valid_uins = set()

# Initialize text splitter with overlap
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", "(?<=\. )", " ", ""]
)

def extract_text_from_pdf(file_bytes: bytes) -> str:
    reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    text = ""
    for page in reader.pages:
        text += page.extract_text() + " "
    return text

@app.post("/upload-pdf")
async def pdf_upload(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a PDF.")
    
    file_bytes = await file.read()
    text = extract_text_from_pdf(file_bytes)
    print("extracted text is ",text)
    
    # Split text with overlap using recursive splitter
    chunks = text_splitter.split_text(text)
    
    # Generate unique identifier
    uin = str(uuid.uuid4())
    
    # Create metadata for each chunk
    metadatas = [{"uin": uin} for _ in chunks]
    
    # Add to vectorstore
    global vectorstore
    if vectorstore is None:
        vectorstore = FAISS.from_texts(chunks, embeddings, metadatas=metadatas)
    else:
        vectorstore.add_texts(chunks, metadatas=metadatas)
    
    valid_uins.add(uin)
    return {"uin": uin, "num_chunks": len(chunks)}

@app.post("/query")
async def query_endpoint(query: str = Body(...),
                         uins: List[str] = Body(...)):
    if not uins:
        raise HTTPException(status_code=400, detail="No UINs provided")
    
    # Validate UINs
    for uin in uins:
        if uin not in valid_uins:
            raise HTTPException(status_code=404, detail=f"UIN {uin} not found")
    
    answers_by_uin = {}
    
    # For each UIN, perform a similarity search and generate an answer
    for uin in uins:
        # Retrieve top 10 chunks only for this specific uin
        docs = vectorstore.similarity_search(
            query, 
            k=5,
            filter=lambda x: x["uin"] == uin
        )
        # print("docs found ",docs)
        if not docs:
            answers_by_uin[uin] = "No relevant information found in provided document."
        else:
            # Build context for this uin
            context = "\n\n".join([doc.page_content for doc in docs])
            # print("Context - ",context)
            
            prompt = PromptTemplate(input_variables=["context", "query"], template=prompt_template)


            formatted_prompt = prompt.format(context=context, query=query)
            answer = llm.invoke(query)
            answers_by_uin[uin] = answer.content
    
    # Merge individual answers into a final composite answer
    final_answer = "\n\n".join(
    [f"{answers_by_uin[uin]}" for uin in uins]
)
    
    return {
        "final_answer": final_answer,
        "individual_answers": answers_by_uin
    }
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)