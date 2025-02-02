import io
import uuid
import PyPDF2
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.schema import StrOutputParser
from langchain.prompts import PromptTemplate

# TruLens Evaluation imports
from trulens_eval import TruChain, Feedback, OpenAI, Tru

# Guardrails for filtering responses

# Load environment variables (make sure you have your .env file with API keys)
load_dotenv()

# Initialize Guardrails for safety filtering

# Initialize OpenAI API provider for evaluation
openai_provider = OpenAI()
tru = Tru()

# Initialize the ChatOpenAI model
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# Define the prompt template for our RAG chain
prompt_template = """
You are an AI assistant that strictly answers questions based on the provided insurance policy document.
Follow these steps to generate an accurate response:

1. Review the past conversation for context.
2. Carefully analyze the user's latest question.
3. Extract only the necessary information from the insurance policy.
4. Do NOT provide financial, investment, or legal advice.

### Past Conversation:  
{chat_history}

### Context from PDF:  
{context}  

### User's Latest Question:  
{question}  

Answer:
"""

# Global dictionary to store processed PDFs keyed by a unique identifier (uin)
pdf_stores = {}

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware to allow requests from any origin (adjust for production)
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace ["*"] with a list of allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Endpoint: /upload-pdf
# ---------------------------
@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF file, extract its text, split it into chunks, 
    create a vectorstore using embeddings, and return a unique identifier (uin).
    """
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a PDF.")
    
    file_bytes = await file.read()
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        pdf_text = "\n".join(
            page.extract_text() for page in pdf_reader.pages if page.extract_text()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error processing PDF file.")
    
    # Split the text into smaller chunks for efficient retrieval
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    documents = text_splitter.create_documents([pdf_text])
    
    # Create vector embeddings and build the vectorstore using FAISS
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)
    
    # Generate a unique identifier for this PDF upload
    uin = str(uuid.uuid4())
    pdf_stores[uin] = {"vectorstore": vectorstore, "chat_history": []}
    print(pdf_stores)
    
    return {"uin": uin}

# ---------------------------
# Endpoint: /query
# ---------------------------
from pydantic import BaseModel

class QueryRequest(BaseModel):
    uin: str
    question: str
    
@app.post("/query")
async def query_pdf(query: QueryRequest):
    """
    Query an uploaded PDF using its unique identifier (uin) and a question.
    The endpoint retrieves the relevant PDF content from the vectorstore, 
    constructs a prompt (including previous conversation context), and generates an answer.
    """
    
    uin = query.uin
    question = query.question
    print("UIN number is ", uin)
    print("Asked question is ", question)
    if uin not in pdf_stores:
        raise HTTPException(status_code=404, detail="PDF not found for the given uin.")
    
    # Retrieve stored vectorstore and chat history for this PDF
    store = pdf_stores[uin]
    vectorstore = store["vectorstore"]
    chat_history = "\n".join(store["chat_history"])
    
    print("till here ok")
    
    # Retrieve top 5 relevant document chunks from the vectorstore
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    docs = retriever.get_relevant_documents(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    
    # Format the prompt using the template, incorporating context, history, and the new question
    formatted_prompt = prompt_template.format(
        chat_history=chat_history, context=context, question=question
    )
    
    # Generate the raw response using the LLM
    raw_response = llm(formatted_prompt)
    
    print("raw response ",raw_response)
    
    # Apply Guardrails filtering for safety
    # guard_filter = guard.to_runnable()
    # safe_response = guard_filter(raw_response)
    
    # Parse the filtered response to produce a final string answer
    output_parser = StrOutputParser()
    parsed_response = output_parser.parse(raw_response)
    
    # Update chat history for future queries
    store["chat_history"].append(f"User: {question}")
    store["chat_history"].append(f"Assistant: {parsed_response}")
    
    return {"reply": parsed_response}

# ---------------------------
# Endpoint: /dashboard
# ---------------------------
@app.get("/dashboard")
async def run_dashboard():
    """
    Launch the TruLens evaluation dashboard.
    """
    tru.run_dashboard()
    return JSONResponse(content={"detail": "TruLens dashboard launched."})

# ---------------------------
# Run the application
# ---------------------------
if __name__ == "__main__":
    import uvicorn
    # Adjust the module path if your file name is different or if the app is in a subdirectory.
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
