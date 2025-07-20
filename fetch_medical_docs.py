import requests
from xml.etree import ElementTree
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from pinecone_setup import get_index

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def search_pubmed(query, max_results=5):
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": f"{query}[Title/Abstract]",
        "retmode": "json",
        "retmax": max_results
    }
    res = requests.get(url, params=params).json()
    return res['esearchresult']['idlist']

def fetch_abstracts(pmids):
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {"db": "pubmed", "id": ",".join(pmids), "retmode": "xml"}
    res = requests.get(url, params=params)
    root = ElementTree.fromstring(res.content)

    articles = []
    for article in root.findall(".//PubmedArticle"):
        title = article.findtext(".//ArticleTitle")
        abstract = article.findtext(".//AbstractText")
        if title and abstract:
            articles.append({"title": title, "abstract": abstract})
    return articles

def embed_articles_to_pinecone(articles):
    docs = []
    seen_titles = set()

    for a in articles:
        if a["title"] not in seen_titles:  # Avoid duplicates
            docs.append(Document(page_content=a["abstract"], metadata={"title": a["title"]}))
            seen_titles.add(a["title"])

    index = get_index()
    vectorstore = PineconeVectorStore(index=index, embedding=embedding_model)
    vectorstore.add_documents(docs)
