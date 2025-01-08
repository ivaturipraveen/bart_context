from flask import Flask, request, jsonify
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
import openai

# Initialize Flask app
app = Flask(__name__)

# OpenAI API key
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
sorted_indices = np.argsort(scores)[::-1] # Sort in descending order
reranked_texts = [top_texts[i] for i in sorted_indices]
return reranked_texts, scores[sorted_indices]

# Generate response with OpenAI
def generate_response_with_openai(query, top_texts, top_metadata=None, history=None, max_texts=3):
top_texts = top_texts[:max_texts]
if top_metadata:
top_metadata = top_metadata[:max_texts]

# Construct the prompt
prompt = (
    "You are an intelligent assistant trained to provide structured, concise, and contextually accurate answers. "
    "Maintain continuity by considering the user's conversation history, if provided. Based on the given history and information, "
    "provide a clear and relevant response to the current query. Include the source of information if available.\n\n"
    "### Additional Instructions ###\n"
    "1. Ensure all answers are presented strictly in the form of bullet points.\n"
    "2. Focus on clarity and brevity while ensuring the response is comprehensive and directly addresses the query.\n"
    "3. Avoid including suggestions or offering further assistance unless explicitly requested by the user.\n"
    "4. Ensure the response is strictly humanized, avoiding overly robotic or generic phrasing.\n"
    "5. Ensure that the result is strictly humanized and written in American English format.\n"
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
    "4. Include the source of the information when presenting the response.\n"
    "5. Avoid using apologetic language like 'I’m sorry.' Instead, confidently provide the best available information.\n"
    "6. Avoid making suggestions, providing additional context, or offering help unless explicitly required.\n"
    "7. Ensure the response is humanized and adheres to American English conventions.\n"
    "8. Format the output in Markdown with elements such as headings, bullet points, or tables.\n"
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
embedding = model.encode([query]).astype('float32') # Use the entire query as a single embedding
return embedding

# Format the sources to include only the PDF name and URL
def format_sources(metadata):
formatted_sources = []
for meta in metadata:
if 'pdf_name' in meta and 'pdf_url' in meta:
formatted_sources.append(f"{meta['pdf_name']}: {meta['pdf_url']}")
return formatted_sources

# Clean up the final response by removing unnecessary text
def clean_final_response(response):
if "Please note" in response:
response = response.split("Please note")[0].strip()
return response

# Deduplicate metadata
def deduplicate_metadata(metadata):
seen = set()
unique = []
for meta in metadata:
meta_key = tuple(sorted(meta.items())) # Create a unique key for each dictionary
if meta_key not in seen:
seen.add(meta_key)
unique.append(meta)
return unique


# Load resources
output_dir = './embeddings_output'
embedding_file = f"{output_dir}/combined_embeddings.npy"
faiss_index_file = f"{output_dir}/combined_faiss.index"
data_file = f"{output_dir}/processed_data.json"
embeddings, index, texts, metadata = load_faiss_and_data(embedding_file, faiss_index_file, data_file)
model = SentenceTransformer('all-MiniLM-L6-v2')

@app.route('/query', methods=['POST'])
def query_faiss():
try:
data = request.json
query = data['query']
history = data.get('history', [])
top_k = data.get('top_k', 4)

# Generate a combined embedding for the query
query_embedding = get_combined_query_embedding(query, model)

# Search FAISS for the top results
distances, indices = index.search(query_embedding, top_k)
top_texts = [texts[idx] for idx in indices[0]]
top_metadata = [metadata[idx] for idx in indices[0]]

# Perform exact keyword matching to refine results
refined_texts = [text for text in top_texts if all(keyword.lower() in text.lower() for keyword in query.split())]

if not refined_texts:
refined_texts = top_texts # Fallback to FAISS results if no exact matches

# Rerank results
reranked_texts, _ = rerank_results(query, refined_texts)
final_response = generate_response_with_openai(query, reranked_texts, history=history)

# Clean the response
final_response = clean_final_response(final_response)

# Deduplicate metadata
unique_metadata = deduplicate_metadata(top_metadata)

# Format sources and remove duplicates
formatted_sources = list(dict.fromkeys(format_sources(unique_metadata[:2])))

return jsonify({
"query": query,
"response": final_response,
"sources": formatted_sources
})

except Exception as e:
return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
app.run(host='0.0.0.0', port=5000)

