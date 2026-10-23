from core.chunker import extract_text_from_pdf, chunk_text
from core.gemini_client import get_embeddings

# 1. Extract and chunk the assignment PDF
text = extract_text_from_pdf("sample.pdf")
chunks = chunk_text(text, chunk_size=50, overlap=10)
print(f"Generated {len(chunks)} chunks.")

# 2. Ping the Gemini API
print("Hitting the embedding endpoint...")
vectors = get_embeddings(chunks[:2])  # Just testing the first two chunks to save time

print(f"Success! Received {len(vectors)} vectors.")
print(f"Vector dimensions: {len(vectors[0])}")