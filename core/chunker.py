import PyPDF2

def extract_text_from_pdf(pdf_path):
    reader = PyPDF2.PdfReader(pdf_path)
    text = ""
    with open(pdf_path, 'rb') as file:
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text.replace('\n', ' ') + " "
        return text

def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []

    Stride = chunk_size - overlap
    for i in range(0, len(words), Stride):
        chunk_text = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk_text)

        if i + chunk_size >= len(words):
            break

    return chunks

'''Following functions are for testing purposes only. They will be removed in the final version.
                                                                                        ~ Prabal
'''
if __name__ == "__main__":
    text = extract_text_from_pdf("sample.pdf")
    chunks = chunk_text(text)

    print(f"Total chunks created: {len(chunks)}")
    print(f"Preview Chunk 1: {chunks[0][:100]}...")