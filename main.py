import PyPDF2
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import openai

# Set up OpenAI API key
openai.api_key = "Your API-Key"

# Load pre-trained embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

# Function to chunk text into manageable pieces
def chunk_text(text, chunk_size=500):
    words = text.split()
    chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

# Function to create embeddings and store in FAISS index
def create_faiss_index(chunks):
    embeddings = embedding_model.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index, embeddings

# Function to perform similarity search
def retrieve_relevant_chunks(query, index, chunks):
    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(np.array(query_embedding), k=5)  # Top 5 results
    relevant_chunks = [chunks[i] for i in indices[0]]
    return relevant_chunks

# Function to generate a response using OpenAI
def generate_response(relevant_chunks, query):
    context = "\n".join(relevant_chunks)
    prompt = f"Using the following information, answer the question:\n\n{context}\n\nQuestion: {query}\nAnswer:"
    try:
        response = openai.Completion.create(
            engine="gpt-3.5-turbo",
            prompt=prompt,
            max_tokens=150
        )
        return response.choices[0].text.strip()
    except Exception as e:
        return f"Error generating response: {str(e)}"

# Main pipeline
def main_pipeline(pdf_path, query):
    # Step 1: Extract text from PDF
    text = extract_text_from_pdf(pdf_path)

    # Step 2: Chunk the text
    chunks = chunk_text(text)

    # Step 3: Create FAISS index
    index, _ = create_faiss_index(chunks)

    # Step 4: Retrieve relevant chunks
    relevant_chunks = retrieve_relevant_chunks(query, index, chunks)

    # Step 5: Generate response
    response = generate_response(relevant_chunks, query)
    return response

# Example usage
if __name__ == "__main__":
    pdf_path ="C:\\Users\\user\\Python Scripts\\.venv\\DSA\\Example.pdf"  # Replace with your PDF file path
    query = "What is the unemployment information based on the type of degree?"
    answer = main_pipeline(pdf_path, query)
    print("Answer:", answer)
