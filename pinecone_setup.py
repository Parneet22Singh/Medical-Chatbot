import os
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

api_key = os.getenv("PINECONE_API_KEY")
host = os.getenv("PINECONE_HOST")

pc = Pinecone(api_key=api_key)
index_name = "medical-chatbot"

def get_index():
    if index_name not in pc.list_indexes().names():
        raise ValueError(f"Index '{index_name}' not found.")
    return pc.Index(host=host)

### NOTE- This program only connects to an existing database named medical-chatbot, so you would have to manually create one for it to connect to.
