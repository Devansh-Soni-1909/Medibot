
# 🩺 Medibot – AI-Powered Symptom-to-Specialist Recommender

**Medibot** is a GenAI-powered medical assistant that helps users determine which **medical department** (e.g., Cardiology, Neurology, Dermatology) they should consult based on their symptoms.  
It uses **Retrieval-Augmented Generation (RAG)** to ensure answers are derived from a trusted medical textbook and not hallucinated by the model.

---

## 🚀 Features

- ✅ Takes symptom input via a Streamlit chatbot interface
- ✅ Uses **Hugging Face LLM (Mistral-7B-Instruct)** via API
- ✅ Retrieves answers from an embedded **medical PDF knowledge base**
- ✅ Displays source excerpts for transparency
- ✅ Designed to reduce misdirected medical appointments

---

## 🧠 How It Works

1. **Medical Knowledge Loading**  
   Loads a medical diagnostic PDF (e.g., *Symptom to Diagnosis*) and splits it into semantic chunks.

2. **Vector Embedding Creation**  
   Uses `all-MiniLM-L6-v2` sentence transformer to embed chunks and stores them in a **FAISS vector database**.

3. **Streamlit Chatbot Interface**  
   Accepts symptom queries from the user and retrieves relevant medical context.

4. **LLM Reasoning with Context**  
   LLM (Mistral-7B-Instruct via HuggingFace Endpoint) is prompted to suggest a suitable department, strictly based on the retrieved medical data.

5. **Transparent Recommendations**  
   The user receives:
   - Department recommendation
   - Source excerpts from the textbook used for the decision

---

## 🛠️ Tech Stack

| Component      | Tool/Library                                |
|----------------|---------------------------------------------|
| Frontend       | [Streamlit](https://streamlit.io)           |
| LLM            | [Mistral-7B-Instruct](https://huggingface.co/mistralai) via HuggingFace Endpoint |
| RAG Framework  | [LangChain](https://www.langchain.com)      |
| Embeddings     | `sentence-transformers/all-MiniLM-L6-v2`    |
| Vector Store   | [FAISS](https://github.com/facebookresearch/faiss) |
| PDF Parser     | LangChain PyPDFLoader                       |
| Environment    | Python (.env for tokens)                    |

---

## 💻 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/Devansh-Soni-1909/medibot.git
cd medibot
```

### 2. Create `.env` file

```env
HF_TOKEN=your_huggingface_api_token_here
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Load medical PDF and create embeddings

```bash
python create_memory_for_llm.py
```

> Make sure your `data/` folder contains your medical reference book in PDF format.

### 5. Run the Streamlit chatbot

```bash
streamlit run medibot.py
```



## 🧪 Example Usage

**User:**  
`Describe your symptoms: chest pain and shortness of breath`

**Medibot:**  
```
Based on your symptoms, I recommend you consult a Pulmonologist or Cardiologist. Possible causes include pneumonia, pleural effusion, or cardiac conditions like pericarditis.

Source: Symptom to Diagnosis, Chapter 10
```

---

## 🔗 Project Links

- 📂 [Project GitHub Repository](https://github.com/Devansh-Soni-1909/medibot)
- 📘 Medical Guidebook used: *Symptom to Diagnosis – An Evidence Based Guide*

---

## ⚠️ Disclaimer

Medibot is for **educational and prototyping purposes only**. It does not provide real medical advice. Always consult a licensed physician for medical concerns.

---

## 🙌 Acknowledgements

- IBM x AICTE GenAI Challenge
- LangChain Team
- Hugging Face
- McGraw-Hill Medical Textbook for reference data
