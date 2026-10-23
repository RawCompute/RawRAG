import os
from google import genai
from dotenv import load_dotenv

load_dotenv()
client = genai.Client()

def get_embeddings(text_chunks):
    """
    Converts a list of text chunks into floating-point vectors in a single batch call.
    """
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text_chunks
    )
    # Extract the raw floating-point arrays directly
    return [emb.values for emb in response.embeddings]

def get_query_embedding(query):
    """
    Embeds the user's search query to compare against the database.
    """
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=query
    )
    return response.embeddings[0].values

def generate_answer(query, context_chunks):
    """
    Injects the retrieved chunks into a hardcoded prompt template 
    and pings the LLM for the final synthesis.
    """
    context_string = "\n\n---\n\n".join(context_chunks)
    
    prompt = f"""
    Answer the following query using strictly the context provided below. 
    If the context does not contain the answer, clearly state that you do not have enough information.
    
    Context:
    {context_string}
    
    Query: {query}
    """
    
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    
    return response.text