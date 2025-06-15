import streamlit as st
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import requests
from googletrans import Translator
import os

# 🔐 Replace with your actual Hugging Face API token
HF_API_KEY = os.getenv("HF_API_KEY")

# 📄 Extract text from uploaded PDF
def extract_text_from_pdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# ✂️ Split text into chunks
def split_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_text(text)

# 🔢 Generate embeddings
def get_embeddings(chunks, model):
    embeddings = model.encode(chunks)
    return np.array(embeddings)

# 📦 Create FAISS index
def create_faiss_index(embeddings):
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index

# 🔍 Search similar chunks
def search_index(query, index, chunks, model):
    query_embedding = model.encode([query])
    D, I = index.search(query_embedding, k=3)
    results = [chunks[i] for i in I[0]]
    return "\n".join(results)

# 💬 Ask LLM using a supported model
def ask_llm(context, question):
    prompt = f"Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    response = requests.post(
        "https://api-inference.huggingface.co/models/tiiuae/falcon-rw-1b",
        headers=headers,
        json={"inputs": prompt}
    )

    try:
        return response.json()[0]['generated_text'].split("Answer:")[-1].strip()
    except Exception as e:
        return f"❌ LLM error: {str(e)}"

# 📋 Extract and summarize clauses (uses same model)
def extract_clauses_with_summary(text):
    prompt = f"""
You are a legal assistant. Extract and summarize key clauses from the contract below.

For each clause, show:
1. Clause Title
2. Original Text
3. Simple Summary

Text:
{text[:3000]}
    """
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    response = requests.post(
        "https://api-inference.huggingface.co/models/tiiuae/falcon-rw-1b",
        headers=headers,
        json={"inputs": prompt}
    )
    try:
        return response.json()[0]['generated_text']
    except Exception as e:
        return f"❌ Clause summary failed: {str(e)}"

# 🌐 Translate answer
def translate_text(text, dest_lang="hi"):
    try:
        translator = Translator()
        result = translator.translate(text, dest=dest_lang)
        return result.text
    except:
        return "❌ Translation failed."

# 🆚 Compare two documents
def compare_documents(text1, text2):
    prompt = f"""
Compare the following two legal documents.

Document A:
{text1[:3000]}

Document B:
{text2[:3000]}

Output:
- Similar clauses
- Key differences
- Summary of both
    """
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    response = requests.post(
        "https://api-inference.huggingface.co/models/tiiuae/falcon-rw-1b",
        headers=headers,
        json={"inputs": prompt}
    )
    try:
        return response.json()[0]['generated_text']
    except Exception as e:
        return f"❌ Comparison failed: {str(e)}"

# 🚀 Streamlit App UI
st.set_page_config(page_title="LegalGPT", layout="wide")
st.title("📄 LegalGPT – AI Legal Document Assistant")

# Load sentence transformer
model = SentenceTransformer('all-MiniLM-L6-v2')

tab1, tab2 = st.tabs(["📑 Single Document", "🆚 Compare Documents"])

# 📑 Tab 1 – Single Document Analysis
with tab1:
    uploaded_file = st.file_uploader("Upload a legal PDF", type="pdf")
    if uploaded_file:
        st.success("✅ PDF uploaded successfully!")
        text = extract_text_from_pdf(uploaded_file)
        chunks = split_text(text)
        embeddings = get_embeddings(chunks, model)
        index = create_faiss_index(embeddings)

        question = st.text_input("🔍 Ask something about this document:")
        if question:
            context = search_index(question, index, chunks, model)
            answer = ask_llm(context, question)
            st.subheader("💬 Answer")
            st.write(answer)

            lang = st.selectbox("🌐 Translate answer to:", ["None", "Hindi", "Tamil"])
            if lang != "None":
                code = "hi" if lang == "Hindi" else "ta"
                translated = translate_text(answer, dest_lang=code)
                st.subheader(f"🈯 Translation ({lang})")
                st.write(translated)

        if st.button("🧾 Extract & Summarize Clauses"):
            clause_summary = extract_clauses_with_summary(text)
            st.subheader("📌 Key Clauses & Summaries")
            st.write(clause_summary)

# 🆚 Tab 2 – Compare Two PDFs
with tab2:
    pdf1 = st.file_uploader("Upload First PDF", type="pdf", key="pdf1")
    pdf2 = st.file_uploader("Upload Second PDF", type="pdf", key="pdf2")

    if pdf1 and pdf2:
        text1 = extract_text_from_pdf(pdf1)
        text2 = extract_text_from_pdf(pdf2)
        comparison = compare_documents(text1, text2)
        st.subheader("📊 Comparison Results")
        st.write(comparison)
