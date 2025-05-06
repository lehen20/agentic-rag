from langchain_ollama import ChatOllama
import yaml

with open("config/settings.yml") as f:
    config = yaml.safe_load(f)

def get_llm():
    llm_cfg = config["llm"]
    return ChatOllama(
        model=llm_cfg["model"],
        temperature=llm_cfg["temperature"]
    )
