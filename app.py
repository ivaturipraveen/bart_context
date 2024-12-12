from flask import Flask, request, jsonify
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
import openai
import os

# Initialize Flask app
app = Flask(__name__)

# Set OpenAI API key
openai.api_key = os.environ.get("OPENAI_API_KEY")

# Load FAISS index and metadata
def load_faiss_and_data(embedding_file, faiss_index_file, data_file):
    embeddings = np.load(embedding_file)
    index = faiss.read_index(faiss_index_file)
    with open(data_file, 'r') as f:
        data = json.load(f)
    return embeddings, index, data["texts"], data["metadata"]

# Rerank results with CrossEncoder
def rerank_results(query, top_texts):
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    scores = cross_encoder.predict([(query, text) for text in top_texts])
    sorted_indices = np.argsort(scores)[::-1]  # Sort in descending order
    reranked_texts = [top_texts[i] for i in sorted_indices]
    return reranked_texts, scores[sorted_indices]

# Generate response with OpenAI
def generate_response_with_openai(query, top_texts, history=None, max_texts=3):
    top_texts = top_texts[:max_texts]  # Limit the number of texts to 3

    # Construct the prompt
    prompt = (
        "You are an intelligent assistant trained to provide structured, concise, and contextually accurate answers. "
        "Maintain continuity by considering the user's conversation history, if provided. Based on the given history and information, "
        "provide a clear and relevant response to the current query. Avoid apologetic language such as 'I'm sorry' and instead offer "
        "helpful context, approximate estimates, or advice when exact information isn't available.\n\n"
    )

    if history:
        prompt += "### Conversation History ###\n"
        prompt += "\n".join([f"User: {h['user']}\nAssistant: {h['assistant']}" for h in history]) + "\n\n"

    prompt += f"### Current Query ###\nUser: {query}\n\n"
    prompt += "### Relevant Information ###\n"
    prompt += "\n".join([f"{i+1}. {text}" for i, text in enumerate(top_texts)]) + "\n\n"
    prompt += (
        "### Instructions for Assistant ###\n"
        "1. Use the conversation history to maintain continuity in your response.\n"
        "2. Summarize the relevant information provided, if applicable.\n"
        "3. Answer the current query clearly and concisely, ensuring the response is accurate and relevant.\n"
        "4. When appropriate, provide additional context or clarification to make your response more complete.\n"
        "5. Avoid using apologetic language like 'I’m sorry'. Instead, confidently provide the best available information.\n"
    )

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4"
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200  # Limit token size for concise answers
        )
        return response.choices[0].message["content"]
    except Exception as e:
        return f"Error generating response: {e}"

# Endpoint for querying the system
@app.route("/query", methods=["POST"])
def query():
    data = request.json
    query = data.get("query")
    history = data.get("history", [])

    if not query:
        return jsonify({"error": "Query is required."}), 400

    # File paths
    output_dir = './embeddings_output'
    embedding_file = f"{output_dir}/combined_embeddings.npy"
    faiss_index_file = f"{output_dir}/combined_faiss.index"
    data_file = f"{output_dir}/processed_data.json"

    # Load data and index
    embeddings, index, texts, metadata = load_faiss_and_data(
        embedding_file, faiss_index_file, data_file
    )
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Search and rerank
    query_embedding = model.encode([query]).astype('float32')
    distances, indices = index.search(query_embedding, 5)  # Always fetch top 5 results
    top_texts = [texts[idx] for idx in indices[0]]

    reranked_texts, _ = rerank_results(query, top_texts)  # Rerank the top 5
    final_response = generate_response_with_openai(query, reranked_texts, history=history)  # Use top 3 reranked texts

    return jsonify({"query": query, "response": final_response})

# Main function to run the app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
