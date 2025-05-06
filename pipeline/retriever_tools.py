from langchain.tools.retriever import create_retriever_tool
from langgraph.prebuilt import ToolNode

def build_retriever_tool(retriever):
    retriever_tool = create_retriever_tool(
        retriever,
        "get_sql_query_from_pdf",
        "Generate an SQL query based on the user's question and database schema"
    )
    return [retriever_tool]
