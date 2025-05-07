from db.lancedb_client import get_lancedb_table
from src.pdf_loader import load_and_split_pdfs
from src.embeddings import get_embeddings
from pipeline.graph import build_graph
from pipeline.retriever_tools import build_retriever_tool
import gradio as gr
from langchain_community.vectorstores import LanceDB


def process_message(user_input):
    docs = load_and_split_pdfs()
    table, db = get_lancedb_table()
    embeddings = get_embeddings()

    vectorstore = LanceDB.from_documents(connection=db, table_name="docs", documents=docs, embedding=embeddings,)
    
    retriever = vectorstore.as_retriever()
    retriever_tool = build_retriever_tool(retriever)
    graph = build_graph(retriever_tool)

    inputs = {"messages": [("user", user_input)]}
    content_output = None

    for output in graph.stream(inputs):
        print("[DEBUG] Step output:", output)
        
        if "generate" in output and "messages" in output["generate"]:
            messages = output["generate"]["messages"]
            content_output = messages[0] if messages else None

    return content_output or "No relevant output found."

iface = gr.Interface(
    fn=process_message,
    inputs="text",
    outputs="text",
    title="Agentic RAG",
    description="Query your PDFs using agentic RAG",
)

iface.launch(debug=True)


# Two APIs: 