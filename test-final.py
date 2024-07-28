import re
import streamlit as st
import pandas as pd
import requests
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from spellchecker import SpellChecker

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize the spell checker
spell = SpellChecker()

# Predefined textual data
documents = [
    "Age: 25, Gender: Male, Relationship: Friend, Occasion: Birthday, Budget: $20, MaxBudget: $30, Gift: Book, Rating: 4.5, Link: example.com/book, Image Link: example.com/book.jpg, Interest: Reading",
    "Age: 30, Gender: Female, Relationship: Sister, Occasion: Graduation, Budget: $50, MaxBudget: $70, Gift: Necklace, Rating: 5.0, Link: example.com/necklace, Image Link: example.com/necklace.jpg, Interest: Jewelry",
    "Age: 35, Gender: Male, Relationship: Colleague, Occasion: Promotion, Budget: $40, MaxBudget: $60, Gift: Pen, Rating: 4.0, Link: example.com/pen, Image Link: example.com/pen.jpg, Interest: Stationery",
    "Age: 28, Gender: Female, Relationship: Girlfriend, Occasion: Anniversary, Budget: $100, MaxBudget: $150, Gift: Perfume, Rating: 4.8, Link: example.com/perfume, Image Link: example.com/perfume.jpg, Interest: Fragrances",
    "Age: 22, Gender: Male, Relationship: Brother, Occasion: Christmas, Budget: $30, MaxBudget: $50, Gift: Headphones, Rating: 4.6, Link: example.com/headphones, Image Link: example.com/headphones.jpg, Interest: Music"
]

# Fetch data from Fake Store API
def fetch_fake_store_data(limit=1000):
    response = requests.get(f'https://fakestoreapi.com/products?limit={limit}')
    if response.status_code == 200:
        return response.json()
    else:
        st.error("Failed to fetch data from Fake Store API")
        return []

# Helper function to preprocess and clean user input to handle noise
def preprocess_user_input(user_input):
    # Convert to lowercase
    user_input = user_input.lower()
    # Remove punctuation
    user_input = re.sub(r'[^\w\s]', '', user_input)
    # Split into words for spell checking
    words = user_input.split()
    # Correct each word if it's misspelled
    corrected_words = [spell.correction(word) if word not in spell else word for word in words]
    # Join the corrected words back into a single string
    return ' '.join(corrected_words)

# Helper function to preprocess the CSV file and clean the data
def preprocess_csv(file_path):
    df = pd.read_csv(file_path)
    df.fillna("", inplace=True)  # Handle missing data by filling with empty string
    df['text'] = df.apply(lambda row: f"Age: {row['Age']}, Gender: {row['Gender']}, "
                                      f"Relationship: {row['Relationship']}, Occasion: {row['Occasion']}, "
                                      f"Budget: {row['Budget']}, MaxBudget: {row['MaxBudget']}, "
                                      f"Gift: {row['Gift']}, Rating: {row['Rating']}, "
                                      f"Link: {row['Link']}, Image Link: {row['Image Link']}, "
                                      f"Interest: {row['Interest']}", axis=1)
    return df['text'].tolist()

# Helper function to split text into chunks
def get_text_chunks(texts):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = []
    for text in texts:
        chunks.extend(text_splitter.split_text(text))
    return chunks

# Helper function to store text chunks in a vector store
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Define the conversational chain with noise robustness in the prompt
def get_conversational_chain():
    prompt_template = """
    You are an AI assistant providing gift suggestions. Please provide multiple gift ideas based on the given context. Consider the specified age, gender, relationship, occasion, budget, and interests. If specific details are not available or if the input is noisy or unclear, provide general suggestions and ask for clarification if needed.

    Context:
    {context}

    Question:
    {question}

    Please list at least 3-5 gift suggestions.

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Process user input and generate a response
def user_input_handler(user_question):
    # Preprocess the user input to handle noise
    cleaned_question = preprocess_user_input(user_question)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(cleaned_question, k=3)
    
    # Extract the context from each document (assuming they are dictionaries)
    contexts = [doc['text'] for doc in docs if 'text' in doc]
    
    chain = get_conversational_chain()
    response = chain({"input_documents": contexts, "question": cleaned_question}, return_only_outputs=True)
    st.write("Reply: ", response["output_text"])

    # Fetch and display products from Fake Store API
    products = fetch_fake_store_data()
    st.write("Top Products from Fake Store API:")
    for product in products:
        st.write(f"Title: {product['title']}, Price: {product['price']}, Rating: {product['rating']}")
        st.image(product['image'], caption=product['title'])

# Main function to run the Streamlit app
def main():
    st.set_page_config(page_title="AI Gift Suggestions")
    st.header("AI Gift Suggestions")

    user_question = st.text_input("Ask a Question")

    if user_question:
        user_input_handler(user_question)

    with st.sidebar:
        st.title("Menu:")
        if st.button("Process Predefined Dataset"):
            with st.spinner("Processing..."):
                # Preprocess predefined documents
                text_chunks = get_text_chunks(documents)
                # Assume the uploaded CSV file is in the same directory
                csv_file_path = 'dataset.csv'
                csv_texts = preprocess_csv(csv_file_path)
                csv_chunks = get_text_chunks(csv_texts)
                all_chunks = text_chunks + csv_chunks
                get_vector_store(all_chunks)
                st.success("Done")

if __name__ == "__main__":
    main()
