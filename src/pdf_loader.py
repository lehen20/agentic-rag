from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import yaml

with open("config/settings.yml") as f:
    config = yaml.safe_load(f)

def load_and_split_pdfs():
    loader = PyPDFDirectoryLoader(config["pdf"]["folder_path"])
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=100,
        chunk_overlap=50
    )

    splits = splitter.split_documents(docs)
    
    for i, doc in enumerate(splits):
        doc.metadata = {
            "producer": doc.metadata.get("producer", "Unknown"),
            "creator": doc.metadata.get("creator", "Unknown"),
            "creationdate": doc.metadata.get("creationdate", "Unknown"),
            "source": doc.metadata.get("source", "source.pdf"),
            "total_pages": doc.metadata.get("total_pages", 0),
            "page": doc.metadata.get("page", i + 1),
            "page_label": doc.metadata.get("page_label", f"Page {i + 1}"),
        }
    return splits
