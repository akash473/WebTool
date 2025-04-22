import streamlit as st
from bs4 import BeautifulSoup
import requests
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def scrape_url(url):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove unwanted elements
        for element in soup(["script", "style", "nav", "footer", "header", "iframe"]):
            element.decompose()
            
        # Get clean text
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        return '\n'.join(chunk for chunk in chunks if chunk)
        
    except Exception as e:
        st.error(f"Error scraping {url}: {str(e)}")
        return None

def process_text(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(text)

def create_vector_store(chunks):
    try:
        if not os.getenv("OPENAI_API_KEY"):
            st.error("OpenAI API key is missing. Please add it to your environment variables.")
            return None
            
        embeddings = OpenAIEmbeddings()
        return FAISS.from_texts(chunks, embeddings)
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        return None

def answer_question(vector_store, question):
    if not vector_store:
        return "Cannot answer - no content processed"
        
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(temperature=0.3),  # Lower temperature for more factual answers
        chain_type="stuff",
        retriever=vector_store.as_retriever()
    )
    return qa.run(question)

def main():
    st.title("Web Content Q&A Tool")
    
    # API key input (alternative to .env file)
    if not os.getenv("OPENAI_API_KEY"):
        api_key = st.text_input("Enter your OpenAI API key", type="password")
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            st.success("API key saved for this session")
    
    # URL input
    st.header("Step 1: Enter URLs")
    urls = st.text_area("Enter one or more URLs (one per line)", height=100)
    
    if st.button("Process URLs") and os.getenv("OPENAI_API_KEY"):
        if not urls.strip():
            st.warning("Please enter at least one URL")
            return
            
        url_list = [url.strip() for url in urls.split('\n') if url.strip()]
        content = ""
        
        with st.spinner("Processing URLs..."):
            for url in url_list:
                text = scrape_url(url)
                if text:
                    content += text + "\n\n"
            
            if content:
                chunks = process_text(content)
                vector_store = create_vector_store(chunks)
                if vector_store:
                    st.session_state.vector_store = vector_store
                    st.success(f"Processed {len(url_list)} URLs with {len(chunks)} text chunks")
            else:
                st.error("No content could be extracted from the URLs")

    # Question input
    st.header("Step 2: Ask a Question")
    question = st.text_input("Ask a question about the content")
    
    if question and 'vector_store' in st.session_state:
        with st.spinner("Finding answer..."):
            answer = answer_question(st.session_state.vector_store, question)
            st.subheader("Answer")
            st.write(answer)

if __name__ == "__main__":
    main()