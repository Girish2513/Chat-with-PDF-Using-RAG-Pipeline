import PyPDF2
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import openai

openai.api_key = "Your API-Key"

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Data Ingestion
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

# Chunking the Data
def chunk_text(text, chunk_size=500):
    words = text.split()
    chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

# Embedding and Indexing
def create_faiss_index(chunks):
    embeddings = embedding_model.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index, embeddings

# Query Handling
def retrieve_relevant_chunks(query, index, chunks):
    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(np.array(query_embedding), k=5)  # Top 5 results
    relevant_chunks = [chunks[i] for i in indices[0]]
    return relevant_chunks

# Generating Responses
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
    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text)
    index, _ = create_faiss_index(chunks)
    relevant_chunks = retrieve_relevant_chunks(query, index, chunks)
    response = generate_response(relevant_chunks, query)
    return response

if __name__ == "__main__":
    pdf_path ="PDF file path"  
    query = "What is the unemployment information based on the type of degree?"
    answer = main_pipeline(pdf_path, query)
    print("Answer:", answer)
