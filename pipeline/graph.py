from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from typing import Annotated, Sequence, TypedDict
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import tools_condition, ToolNode
# from pipeline.retriever_tools import build_retriever_tool
from pipeline.steps import grade_documents, generate, rewrite
from pipeline.agent import agent

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

def build_graph(retriever_tool):
    graph = StateGraph(AgentState)
    graph.add_node("agent", lambda state: agent(state, retriever_tool))
    graph.add_node("retrieve", ToolNode(retriever_tool))
    graph.add_node("rewrite", rewrite)
    graph.add_node("generate", generate)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", tools_condition, {"tools": "retrieve", END: END})
    graph.add_conditional_edges("retrieve", grade_documents)
    graph.add_edge("generate", END)
    graph.add_edge("rewrite", "agent")
    return graph.compile()
