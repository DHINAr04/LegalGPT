import streamlit as st
import fitz  # PyMuPDF
import os
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from googletrans import Translator
import speech_recognition as sr
import tempfile
import pyttsx3

# 🌐 Load API key
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# 🔊 Initialize Text-to-Speech engine
tts = pyttsx3.init()
tts.setProperty('rate', 160)

# 📄 Extract PDF text
def extract_text_from_pdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    return "".join([page.get_text() for page in doc])

# ✂️ Chunk text
def split_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_text(text)

# 🧠 Embedding & Index
def get_embeddings(chunks, model):
    return np.array(model.encode(chunks))

def create_faiss_index(embeddings):
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index

def search_index(query, index, chunks, model):
    query_embedding = model.encode([query])
    D, I = index.search(query_embedding, k=3)
    return "\n".join([chunks[i] for i in I[0]])

# 💬 Call OpenRouter LLM
def ask_llm(context, question):
    prompt = f"Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "https://your-app.streamlit.app",
        "X-Title": "LegalGPT"
    }
    data = {
        "model": "mistralai/mistral-7b-instruct",  # or gpt-3.5-turbo
        "messages": [
            {"role": "system", "content": "You are a legal assistant."},
            {"role": "user", "content": prompt}
        ]
    }
    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions",
                                 headers=headers, json=data)
        if response.status_code != 200:
            return f"❌ LLM API error ({response.status_code}): {response.text}"
        return response.json()['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"❌ LLM exception: {str(e)}"

# 🌍 Translate
def translate_text(text, dest_lang_code):
    try:
        translator = Translator()
        return translator.translate(text, dest=dest_lang_code).text
    except:
        return "❌ Translation failed."

# 🎤 Voice recognition
def transcribe_audio(audio_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return "❌ Could not understand audio."
    except sr.RequestError:
        return "❌ Voice service error."

# 🔊 Speak answer aloud
def speak_text(text):
    tts.say(text)
    tts.runAndWait()

# 🚀 Streamlit App
st.set_page_config(page_title="LegalGPT with Voice", layout="wide")
st.title("⚖️ LegalGPT – Voice-Powered AI Legal Assistant")

model = SentenceTransformer('all-MiniLM-L6-v2')

tab1, tab2 = st.tabs(["📑 Single Document", "🆚 Compare Documents"])

# === Tab 1: Upload & Ask ===
with tab1:
    uploaded_file = st.file_uploader("Upload a legal PDF", type="pdf")
    if uploaded_file:
        text = extract_text_from_pdf(uploaded_file)
        chunks = split_text(text)
        embeddings = get_embeddings(chunks, model)
        index = create_faiss_index(embeddings)

        st.markdown("### 🎤 Ask via microphone or type a question:")

        audio_data = st.file_uploader("Upload voice (WAV only)", type=["wav"])
        question = st.text_input("Or type your legal question here:")

        if audio_data:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(audio_data.read())
                voice_question = transcribe_audio(tmp.name)
                st.write("🗣️ Transcribed:", voice_question)
                question = voice_question

        if question:
            context = search_index(question, index, chunks, model)
            answer = ask_llm(context, question)
            st.subheader("💬 AI Answer")
            st.write(answer)

            if st.button("🔊 Speak the answer"):
                speak_text(answer)

            lang_options = {
                "Hindi": "hi", "Tamil": "ta", "Telugu": "te",
                "Marathi": "mr", "Bengali": "bn", "French": "fr",
                "German": "de", "Spanish": "es"
            }

            target_lang = st.selectbox("🌍 Translate answer to:", ["None"] + list(lang_options.keys()))
            if target_lang != "None":
                translated = translate_text(answer, lang_options[target_lang])
                st.subheader(f"🈯 Translated ({target_lang})")
                st.write(translated)

# === Tab 2: Compare PDFs ===
with tab2:
    pdf1 = st.file_uploader("Upload First PDF", type="pdf", key="pdf1")
    pdf2 = st.file_uploader("Upload Second PDF", type="pdf", key="pdf2")

    if pdf1 and pdf2:
        text1 = extract_text_from_pdf(pdf1)
        text2 = extract_text_from_pdf(pdf2)
        combined = f"Compare these two contracts:\n\nA:\n{text1[:3000]}\n\nB:\n{text2[:3000]}"
        comparison = ask_llm("", combined)
        st.subheader("📊 AI Comparison")
        st.write(comparison)
