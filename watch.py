import os
import json
from datetime import datetime
from gpt4all import GPT4All
import PyPDF2
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Initialize FastAPI app
app = FastAPI()

# Pydantic model for incoming JSON request
class Message(BaseModel):
    timestamp: str
    sender: str
    from_number: str
    text: str

# Pydantic model for classified message response
class ClassifiedMessage(BaseModel):
    timestamp: str
    sender: str
    from_number: str
    text: str
    category: str
    urgency: str

# Global variables for model and vector store
index = None
chunks = None
embedder = None
model = None
tokenizer = None
CLASSIFIED_LOG_FILE = "classified_messages.log"

# Step 1: Load and preprocess local documents
def load_documents(folder_path: str) -> list:
    """Load text and PDF documents from the specified folder."""
    documents = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if filename.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as f:
                documents.append(f.read())
        elif filename.endswith('.pdf'):
            with open(file_path, 'rb') as f:
                pdf = PyPDF2.PdfReader(f)
                text = ''
                for page in pdf.pages:
                    text += page.extract_text() or ''
                documents.append(text)
    return documents

# Step 2: Split documents into smaller chunks
def chunk_documents(documents: list, chunk_size: int = 256) -> list:
    """Split documents into chunks of specified size."""
    chunks = []
    for doc in documents:
        words = doc.split()
        for i in range(0, len(words), chunk_size):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
    return chunks

# Step 3: Embed documents and create a vector store
def create_vector_store(chunks: list, model_name: str = 'all-MiniLM-L6-v2') -> tuple:
    """Create a FAISS index for document chunks."""
    embedder = SentenceTransformer(model_name)
    embeddings = embedder.encode(chunks, show_progress_bar=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings, dtype='float32'))
    return index, chunks, embedder

# Step 4: Retrieve relevant chunks for a query
def retrieve_chunks(query: str, index, chunks: list, embedder, top_k: int = 2) -> list:
    """Retrieve top-k relevant document chunks for a query."""
    query_embedding = embedder.encode([query])[0]
    distances, indices = index.search(np.array([query_embedding], dtype='float32'), top_k)
    return [chunks[i] for i in indices[0]]

# Step 5: Count tokens using Mistral tokenizer
def count_tokens(text: str, tokenizer) -> int:
    """Count the number of tokens in the text using the tokenizer."""
    return len(tokenizer.encode(text))

# Step 6: Truncate context to fit within token limit
def truncate_context(context: str, query: str, tokenizer, max_tokens: int = 1800) -> str:
    """Truncate context to fit within the token limit."""
    context_tokens = tokenizer.encode(context)
    query_tokens = tokenizer.encode(query)
    total_tokens = len(context_tokens) + len(query_tokens) + 50  # Buffer for prompt
    if total_tokens <= max_tokens:
        return context
    max_context_tokens = max_tokens - len(query_tokens) - 50
    return tokenizer.decode(context_tokens[:max_context_tokens])

# Step 7: Generate response with Nous-Hermes-2-Mistral-7B-DPO
def generate_response(query: str, retrieved_chunks: list, model, tokenizer) -> str:
    """Generate a JSON response classifying the query into category and urgency."""
    context = '\n\n'.join(retrieved_chunks)
    context = truncate_context(context, query, tokenizer)

    prompt = f"""<s>[INST] You are a helpful assistant. Classify the following message into a category (friends, work, Other) and urgency (Low, Medium, High) based on the provided context and provide a reason for your decision. Respond **only** with a JSON object containing the fields 'reason', 'category', and 'urgency'. Do not include any additional text, explanations, or markdown.

Context:
{context}

Message: {query}

Example response:
{{
  "reason": "The message mentions a casual greeting and a personal contact, indicating a social interaction.",
  "category": "friends",
  "urgency": "Low"
}}
[/INST]"""

    prompt_tokens = count_tokens(prompt, tokenizer)
    print(f"Prompt token count: {prompt_tokens}")
    if prompt_tokens > 2048:
        print("Warning: Prompt exceeds 2048 tokens.")
        return json.dumps({
            "reason": "Token limit exceeded",
            "category": "Other",
            "urgency": "Low"
        })

    response = model.generate(prompt, max_tokens=200, temp=0.3, top_p=0.9)
    return response

# Step 8: Classify a message
def classify_message(message: Message) -> dict:
    """Classify a message and return the result as a dictionary."""
    global index, chunks, embedder, model, tokenizer
    if not all([index, chunks, embedder, model, tokenizer]):
        raise HTTPException(status_code=500, detail="Model or vector store not initialized.")

    retrieved_chunks = retrieve_chunks(message.text, index, chunks, embedder)
    response = generate_response(message.text, retrieved_chunks, model, tokenizer)

    try:
        result = json.loads(response)
        category = result.get("category", "Other")
        urgency = result.get("urgency", "Low")
        reason = result.get("reason", "No reason provided")
    except json.JSONDecodeError:
        print("Error: Model did not return valid JSON. Falling back to default.")
        print(result)
        category, urgency, reason = "Other", "Low", "Invalid JSON response from model"

    return {
        "timestamp": message.timestamp,
        "sender": message.sender,
        "from_number": message.from_number,
        "text": message.text,
        "category": category,
        "urgency": urgency,
        "reason": reason  # Include reason for logging/debugging
    }

# FastAPI endpoint to classify a message
@app.post("/classify", response_model=ClassifiedMessage)
async def classify_single_message(message: Message):
    """Classify a single message and log the result."""
    try:
        classified = classify_message(message)
        with open(CLASSIFIED_LOG_FILE, "a") as f:
            f.write(json.dumps(classified) + "\n")
        return ClassifiedMessage(**classified)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error classifying message: {str(e)}")

# Initialize models and vector store at startup
@app.on_event("startup")
async def startup_event():
    """Initialize models and vector store when the FastAPI server starts."""
    global index, chunks, embedder, model, tokenizer
    folder_path = './'  # Replace with your documents folder path
    model_path = 'Nous-Hermes-2-Mistral-7B-DPO.Q4_0.gguf'
    embedding_model = 'all-MiniLM-L6-v2'

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('mistralai/Mixtral-8x7B-Instruct-v0.1')

    print("Loading documents...")
    documents = load_documents(folder_path)
    if not documents:
        print("No documents found in the folder.")
        raise HTTPException(status_code=500, detail="No documents found in the folder.")

    chunks = chunk_documents(documents, chunk_size=256)

    print("Embedding documents...")
    index, chunks, embedder = create_vector_store(chunks, embedding_model)

    print("Loading GPT4All model...")
    model = GPT4All(model_name=model_path, device='gpu' if faiss.get_num_gpus() > 0 else 'cpu')

# Main execution for testing
def main():
    """Run a test classification query."""
    global index, chunks, embedder, model, tokenizer
    folder_path = './'
    model_path = 'Nous-Hermes-2-Mistral-7B-DPO.Q4_0.gguf'
    embedding_model = 'all-MiniLM-L6-v2'

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('mistralai/Mixtral-8x7B-Instruct-v0.1')

    print("Loading documents...")
    documents = load_documents(folder_path)
    if not documents:
        print("No documents found in the folder.")
        return

    chunks = chunk_documents(documents, chunk_size=256)

    print("Embedding documents...")
    index, chunks, embedder = create_vector_store(chunks, embedding_model)

    print("Loading GPT4All model...")
    model = GPT4All(model_name=model_path, device='gpu' if faiss.get_num_gpus() > 0 else 'cpu')

    # Test query
    query = ""
    print(f"Query: {query}")
    retrieved_chunks = retrieve_chunks(query, index, chunks, embedder)
    response = generate_response(query, retrieved_chunks, model, tokenizer)
    print("Response:", response)

if __name__ == '__main__':
    # Run main for testing
    main()
    # Start FastAPI server
    uvicorn.run(app, host="0.0.0.0", port=8000)
