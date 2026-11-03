import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="RawRAG UI", layout="centered")
st.title("RawRAG")

st.sidebar.header("Document Ingestion")
uploaded_file = st.sidebar.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file and st.sidebar.button("Process PDF"):
    with st.sidebar.status("Chunking and Embedding..."):
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
        response = requests.post(f"{API_URL}/ingest", files=files)
        
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "already_ingested":
                st.sidebar.info("Document is already in the database.")
            else:
                st.sidebar.success(f"Ingested {data.get('chunks')} chunks from {data.get('pages')} pages.")
        else:
            st.sidebar.error(f"Error: {response.text}")

st.subheader("Query your Documents")
question = st.text_input("Ask a question about the uploaded PDF:")

if st.button("Search") and question:
    with st.spinner("Running vector search and generating answer..."):
        response = requests.post(
            f"{API_URL}/query", 
            json={"question": question, "top_k": 3}
        )
        
        if response.status_code == 200:
            data = response.json()
            st.markdown("### Answer")
            st.write(data.get("answer", "No answer generated."))
            
            st.markdown("### Sources")
            for source in data.get("sources", []):
                with st.expander(f"Page {source['page']} | Confidence Score: {source['score']}"):
                    st.caption(f"Document: {source['document']}")
                    st.write(source["text"])
        else:
            st.error(f"Backend Error: {response.text}")