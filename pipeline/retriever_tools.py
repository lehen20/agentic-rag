from langchain.tools.retriever import create_retriever_tool
from langgraph.prebuilt import ToolNode


def build_retriever_tool(retriever):
    retriever_tool = create_retriever_tool(
        retriever,
        "get_summary_from_pdf",
        "Generate a summary of the PDF document based on the user's question",
    )
    return [retriever_tool]
