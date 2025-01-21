from flask import Flask, request, jsonify
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
import openai

# Initialize Flask app
app = Flask(__name__)

# Set OpenAI API key
openai.api_key = "YOUR_OPENAI_API_KEY"

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
def generate_response_with_openai(query, top_texts, top_metadata=None, history=None, max_texts=3):
    top_texts = top_texts[:max_texts]
    if top_metadata:
        top_metadata = top_metadata[:max_texts]

    # Construct the prompt
    prompt = (
    "You are the BART Assistant, a specialized AI trained to assist users with information about the Bay Area Rapid Transit (BART). "
    "Your role is to provide accurate, concise, and relevant answers to user queries. "
    "Always refer to yourself as the BART Assistant and avoid mentioning that you are an AI model, ChatGPT, or any other generic identity. "
    "You must always present yourself as a BART-specific assistant.\n\n"
    "Focus on being natural, relatable, and human-like in tone.\n\n"
    "### Instructions ###\n"
    "1. For queries related to your identity, always respond: 'I am the BART Assistant, trained to assist you with information about BART.'\n"
    "2. For all other queries, answer clearly and concisely, focusing on the user's needs.\n"
    "3. Maintain a professional, human-like tone in your responses.\n"
    "4. Answer queries without including the sources in the main content.\n"
    "5. At the end of the response, include sources or references to support your answers under a separate heading: '### Sources'.\n"
    "6. Do not provide information outside the scope of BART-related queries.\n"
    "7. Strictly ensure the output is humanized and adheres to American English conventions in grammar, spelling, and phrasing.\n"
    )

    if history:
        prompt += "### Conversation History ###\n"
        prompt += "\n".join([f"User: {h['user']}\nAssistant: {h['assistant']}" for h in history]) + "\n\n"

    prompt += f"### Current Query ###\nUser: {query}\n\n"
    prompt += "### Relevant Information ###\n"
    for i, text in enumerate(top_texts):
        source_info = f" (Source: {top_metadata[i]})" if top_metadata else ""
        prompt += f"{i+1}. {text}{source_info}\n"

    prompt += (
    "\n### Instructions for Assistant ###\n"
    "1. Use the conversation history to maintain continuity in your response.\n"
    "2. Focus on information that combines all relevant parts of the query.\n"
    "3. Answer the current query clearly and concisely, ensuring the response is accurate and relevant.\n"
    "4. Do not include sources within the main content of the response.\n"
    "5. Instead, provide a separate list of sources at the end under the heading '### Sources'.\n"
    "6. Avoid using apologetic language like 'I’m sorry'. Instead, confidently provide the best available information.\n"
    "7. Avoid making suggestions, providing additional context, or offering help unless explicitly required.\n"
    "8. Strictly ensure the response is humanized and adheres to American English conventions.\n"
    "9. Do not include sources within the content of your response.\n"

    )


    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500
        )
        return response.choices[0].message["content"]
    except Exception as e:
        print(f"Error generating response: {e}")
        return None

# Combine query keywords into a single embedding
def get_combined_query_embedding(query, model):
    embedding = model.encode([query]).astype('float32')  # Use the entire query as a single embedding
    return embedding

# Clean up the final response by removing unnecessary text
def clean_final_response(response):
    # Remove any '### Sources' or 'Sources:' section from the response
    if "### Sources" in response:
        response = response.split("### Sources")[0].strip()
    elif "Sources:" in response:
        response = response.split("Sources:")[0].strip()

    # Remove "Please note" text if present
    if "Please note" in response:
        response = response.split("Please note")[0].strip()
    return response

def format_sources(metadata):
    formatted_sources = []
    for meta in metadata:
        if 'pdf_name' in meta and 'pdf_url' in meta and 'http' in meta['pdf_url']:
            formatted_sources.append(f"{meta['pdf_name']}: {meta['pdf_url']}")
    return formatted_sources

def format_html_response(query, final_response, sources):
    # Clean the response first
    final_response = clean_final_response(final_response)

    # Start building the HTML response
    html_response = f"<h1>Query: {query}</h1>\n"
    html_response += "<h2>Response</h2>\n"
    html_response += f"<p>{final_response}</p>\n"

    # Only add sources if they contain valid URLs
    valid_sources = [source for source in sources if ":" in source and "http" in source]
    if valid_sources:
        html_response += "<h2>Sources</h2>\n<ul>\n"  # Start an unordered list
        for source in valid_sources:
            try:
                pdf_name, pdf_url = source.split(": ", 1)
                html_response += f"  <li><a href='{pdf_url}' target='_blank'>{pdf_name}</a></li>\n"
            except ValueError:
                continue  # Skip malformed sources
        html_response += "</ul>\n"  # Close the unordered list

    return html_response

# Deduplicate metadata
def deduplicate_metadata(metadata):
    seen = set()
    unique = []
    for meta in metadata:
        if 'pdf_url' in meta and 'http' in meta['pdf_url']:
            meta_key = tuple(sorted(meta.items()))
            if meta_key not in seen:
                seen.add(meta_key)
                unique.append(meta)
    return unique

# Test function
def test_faiss_with_reranking_and_generation(query, index, texts, metadata, model, history=None, top_k=4):
    # Generate a combined embedding for the query
    query_embedding = get_combined_query_embedding(query, model)

    # Search FAISS for the top results
    distances, indices = index.search(query_embedding, top_k)
    top_texts = [texts[idx] for idx in indices[0]]
    top_metadata = [metadata[idx] for idx in indices[0]]

    # Perform exact keyword matching to refine results
    refined_texts = [text for text in top_texts if all(keyword.lower() in text.lower() for keyword in query.split())]

    if not refined_texts:
        refined_texts = top_texts  # Fallback to FAISS results if no exact matches

    # Rerank results
    reranked_texts, _ = rerank_results(query, refined_texts)
    final_response = generate_response_with_openai(query, reranked_texts, history=history)

    # Clean the response
    final_response = clean_final_response(final_response)

    # Deduplicate metadata
    unique_metadata = deduplicate_metadata(top_metadata)

    # Format sources and remove duplicates
    formatted_sources = list(dict.fromkeys(format_sources(unique_metadata[:2])))

    formatted_html = format_html_response(query, final_response, formatted_sources)
    return formatted_html

@app.route('/query', methods=['POST'])
def query():
    data = request.get_json()
    query = data.get('query')
    history = data.get('history', [])

    output_dir = './embeddings_output'
    embedding_file = f"{output_dir}/combined_embeddings.npy"
    faiss_index_file = f"{output_dir}/combined_faiss.index"
    data_file = f"{output_dir}/processed_data.json"

    # Load data and index
    embeddings, index, texts, metadata = load_faiss_and_data(
        embedding_file, faiss_index_file, data_file
    )
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Test the system
    result = test_faiss_with_reranking_and_generation(query, index, texts, metadata, model, history=history, top_k=4)

    return jsonify({'response': result})

if __name__ == '__main__':
    app.run(debug=True)
