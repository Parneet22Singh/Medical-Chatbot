import socket
import requests
from xml.etree import ElementTree
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings

# Force IPv4 DNS resolution to avoid flaky IPv6 issues
original_getaddrinfo = socket.getaddrinfo
def getaddrinfo_ipv4_only(host, port, family=0, type=0, proto=0, flags=0):
    return original_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)
socket.getaddrinfo = getaddrinfo_ipv4_only

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def requests_retry_session(
    retries=3,
    backoff_factor=1,
    status_forcelist=(500, 502, 503, 504),
    session=None,
):
    session = session or requests.Session()
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods=frozenset(['GET', 'POST']),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session

def search_pubmed(query, max_results=5):
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": f"{query}[Title/Abstract]",
        "retmode": "json",
        "retmax": max_results,
    }

    session = requests_retry_session()
    try:
        res = session.get(url, params=params, timeout=10)
        res.raise_for_status()
        data = res.json()
        pmids = data.get('esearchresult', {}).get('idlist', [])
        print(f"[search_pubmed] Found {len(pmids)} PMIDs for query '{query}'")
        return pmids
    except requests.exceptions.RequestException as e:
        print(f"[search_pubmed] Error during PubMed search: {e}")
        return []

def fetch_abstracts(pmids):
    if not pmids:
        print("[fetch_abstracts] No PMIDs provided.")
        return []
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {"db": "pubmed", "id": ",".join(pmids), "retmode": "xml"}
    session = requests_retry_session()
    try:
        res = session.get(url, params=params, timeout=10)
        res.raise_for_status()
        root = ElementTree.fromstring(res.content)

        articles = []
        for article in root.findall(".//PubmedArticle"):
            title = article.findtext(".//ArticleTitle")
            # AbstractText can be multiple elements or a single string
            abstract_texts = article.findall(".//AbstractText")
            if abstract_texts:
                abstract = " ".join([at.text for at in abstract_texts if at.text])
            else:
                abstract = None

            if title and abstract:
                articles.append({"title": title, "abstract": abstract})
        print(f"[fetch_abstracts] Fetched {len(articles)} articles with abstracts.")
        return articles
    except requests.exceptions.RequestException as e:
        print(f"[fetch_abstracts] Error fetching abstracts: {e}")
        return []
    except Exception as e:
        print(f"[fetch_abstracts] Parsing error: {e}")
        return []
