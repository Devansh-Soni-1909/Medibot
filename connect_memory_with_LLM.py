import os
from dotenv import load_dotenv, find_dotenv

from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load environment variables
load_dotenv(find_dotenv())
HF_TOKEN = os.environ.get("HF_TOKEN")
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
DB_FAISS_PATH = "vectorstore/db_faiss"

# Step 1: Setup LLM (Mistral with HuggingFace)
def load_llm(huggingface_repo_id):
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        max_new_tokens=512,
        huggingfacehub_api_token=HF_TOKEN
    )
    return llm

# Step 2: Define the custom medical triage prompt
CUSTOM_PROMPT_TEMPLATE = """
You are a medical assistant that helps patients find the right specialist based on their symptoms.

Use the context provided to recommend which medical department (e.g., Cardiology, Dermatology, Neurology, Endocrinology) the user should consult.

If the context doesn't mention symptoms that match the user's case, say:
"I'm not sure which department to suggest based on your symptoms."

Context:
{context}

Symptoms:
{question}

Your Recommendation:
"""

def set_custom_prompt(custom_prompt_template):
    return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

# Step 3: Load the FAISS vector database
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# Step 4: Create RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(HUGGINGFACE_REPO_ID),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k': 3}),
    return_source_documents=True,
    chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

# Step 5: Accept user query and return department recommendation
user_query = input("Describe your symptoms: ")
response = qa_chain.invoke({'query': user_query})

# Output the result and the matching documents
print("\n📋 Recommendation:\n", response["result"])
print("\n📚 Source Documents:\n")
for doc in response["source_documents"]:
    print("—", doc.metadata.get("source", "Unnamed Document"))
    print(doc.page_content)
    print("-" * 50)
