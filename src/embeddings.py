from langchain_huggingface import HuggingFaceEmbeddings
import yaml

with open("config/settings.yml") as f:
    config = yaml.safe_load(f)

def get_embeddings():
    model_cfg = config["embedding"]
    print(f"Loading embeddings with config: {model_cfg}")
    return HuggingFaceEmbeddings(
        model_name=model_cfg["model_name"],
        model_kwargs={"device": model_cfg["device"]}
    )
