from flask import Flask, request, jsonify
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
import openai

app = Flask(__name__)

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
def generate_response_with_openai(query, top_texts, max_texts=3):
    top_texts = top_texts[:max_texts]
    prompt = (
        f"You are an intelligent assistant trained to provide structured, concise, and contextually accurate answers. Based on the following information, summarize the given texts effectively and provide an answer to the query: '{query}'.\n\n"
        + "\n".join([f"{i+1}. {text}" for i, text in enumerate(top_texts)]) +
        "\n\nWhen the information is long or detailed, organize your response in a clear structure, such as bullet points or numbered lists. Ensure the response is directly relevant to the query and avoids introducing any information not present in the provided texts. Maintain clarity and focus in your summary."
    )
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150
        )
        return response.choices[0].message["content"]
    except Exception as e:
        print(f"Error generating response: {e}")
        return None

# Test FAISS with reranking and generative synthesis
def test_faiss_with_reranking_and_generation(query, index, texts, model, top_k=5):
    query_embedding = model.encode([query]).astype('float32')
    distances, indices = index.search(query_embedding, top_k)
    top_texts = [texts[idx] for idx in indices[0]]
    reranked_texts, _ = rerank_results(query, top_texts)
    final_response = generate_response_with_openai(query, reranked_texts)
    return final_response

@app.route('/query', methods=['POST'])
def handle_query():
    data = request.json
    query = data.get("query")
    if not query:
        return jsonify({"error": "Query is required"}), 400

    try:
        output_dir = './embeddings_output'
        embedding_file = f"{output_dir}/combined_embeddings.npy"
        faiss_index_file = f"{output_dir}/combined_faiss.index"
        data_file = f"{output_dir}/processed_data.json"

        embeddings, index, texts, metadata = load_faiss_and_data(
            embedding_file, faiss_index_file, data_file
        )
        model = SentenceTransformer('all-MiniLM-L6-v2')
        answer = test_faiss_with_reranking_and_generation(query, index, texts, model, top_k=5)
        return jsonify({"query": query, "answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
