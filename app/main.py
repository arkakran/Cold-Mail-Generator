import streamlit as st
from langchain_groq import ChatGroq
import chromadb
import pandas as pd
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import uuid
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve the API key from environment variables
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    st.error("API key not found. Please check your .env file.")

# Initialize LLM
llm = ChatGroq(
    temperature=0,
    groq_api_key=groq_api_key,  # Now using the API key from the .env file
    model_name="llama-3.3-70b-versatile"
)

# Load Portfolio Data
df = pd.read_csv("my_portfolio.csv")

# Initialize ChromaDB
client = chromadb.PersistentClient('vectorstore')
collection = client.get_or_create_collection(name="portfolio")

if not collection.count():
    for _, row in df.iterrows():
        collection.add(documents=row["Techstack"],
                       metadatas={"links": row["Links"]},
                       ids=[str(uuid.uuid4())])

# Custom CSS for styling
st.markdown("""
    <style>
        .title {
            text-align: center;
            color: #4CAF50;
            font-size: 3em;
            font-family: 'Arial', sans-serif;
        }
        .subheader {
            text-align: center;
            font-size: 1.2em;
            color: #555;
            margin-bottom: 20px;
        }
        .input-box, .generate-btn {
            display: flex;
            justify-content: center;
            margin-bottom: 20px;
        }
        .email-box {
            background-color: #f9f9f9;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<div class='title'>Cold Email Generator</div>", unsafe_allow_html=True)
st.markdown("<div class='subheader'>Generate professional cold emails effortlessly</div>", unsafe_allow_html=True)

# Input URL
st.markdown("<div class='input-box'>", unsafe_allow_html=True)
url = st.text_input("Enter the job posting URL:")
st.markdown("</div>", unsafe_allow_html=True)

# Button to generate email
st.markdown("<div class='generate-btn'>", unsafe_allow_html=True)
if st.button("Generate Email") and url:
    try:
        # Load webpage content
        loader = WebBaseLoader(url)
        page_data = loader.load().pop().page_content
        
        # Extract job details
        prompt_extract = PromptTemplate.from_template(
            """
            ### SCRAPED TEXT FROM WEBSITE:
            {page_data}
            ### INSTRUCTION:
            Extract the job postings and return a JSON object with the following keys:
            `roles` (list of dicts) with `role`, `experience`, `skills`, and `description`.
            Ensure valid JSON output.
            """
        )
        chain_extract = prompt_extract | llm 
        res = chain_extract.invoke(input={'page_data': page_data})

        json_parser = JsonOutputParser()
        job = json_parser.parse(res.content)

        # Retrieve portfolio links based on job skills
        skills_list = [str(role['skills']) for role in job['roles']]
        links = collection.query(query_texts=skills_list, n_results=2).get('metadatas', [])
        
        # Generate cold email
        prompt_email = PromptTemplate.from_template(
            """
            ### JOB DESCRIPTION:
            {job_description}
            
            ### INSTRUCTION:
            Write a cold email as Mohan, BDE at AtliQ, an AI & Software Consulting company.
            Describe AtliQ's expertise and how it can fulfill the job requirements.
            Include the most relevant links from: {link_list}
            
            ### EMAIL (NO PREAMBLE):
            """
        )
        chain_email = prompt_email | llm
        email_response = chain_email.invoke({"job_description": str(job), "link_list": links})

        # Display generated email
        st.markdown("<div class='email-box'>", unsafe_allow_html=True)
        st.subheader("Generated Cold Email")
        st.text_area("Email Content", email_response.content, height=300)
        st.markdown("</div>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error: {str(e)}")
else:
    st.info("Please enter a valid job posting URL to proceed.")










