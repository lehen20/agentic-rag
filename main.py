from langchain_community.vectorstores import LanceDB
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langgraph.graph.message import add_messages
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import ToolNode
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, StateGraph, START
from langchain_core.documents import Document
from langchain_core.tools import Tool
from langchain_core.messages import BaseMessage

import gradio as gr
from typing import Annotated, Dict, List, Literal, Sequence, TypedDict
from pydantic import BaseModel, Field
import lancedb
import re
import json

from src.redis import get_redis
from src.model import model

# Initialize Redis and model
get_redis()
model = model

# Setup embeddings and document processing
embeddings_mini = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cuda:0"},
)

pdf_folder_path = "docs"
loader = PyPDFDirectoryLoader(pdf_folder_path)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50,
)

documents = text_splitter.split_documents(docs)

for doc in documents:
    if "creationdate" in doc.metadata:
        # print(f"{'creationdate' in doc.metadata=}")
        del doc.metadata["creationdate"]
        # print(f"{'creationdate' in doc.metadata=}")

for doc in documents:
    doc.metadata.pop("creationdate", None)
    # print(f"{'creationdate' in doc.metadata=}")

for doc in documents:
    doc.metadata = {}

# Setup vector database
db = lancedb.connect("/tmp/lancedb")

vectorstore = LanceDB.from_documents(
    documents=documents,
    embedding=embeddings_mini,
)
retriever = vectorstore.as_retriever()

# Define tools
# retriver_tool = create_retreiver_tool(
    
# )
retriever_tool = Tool(
    name="get_sql_query_from_pdf",
    description="Generate an SQL query based on the user's question and database schema",
    func=lambda input: retriever.invoke(input),
)

tools = [retriever_tool]
tool_executor = ToolNode(tools)


# Define state
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    next: str


# LangGraph node functions
def route(
    state: AgentState,
) -> Literal["retrieve", "generate_response", "rewrite_query"]:
    messages = state["messages"]
    print(f"{messages=}")
    last_message = messages[-1]
    
    print(f"{last_message=}")
    if isinstance(last_message, HumanMessage):
        return "retrieve"
        
    # if last_message.role == "user":
    #     return "retrieve"

    return state.get("next", "generate_response")


def retrieve_docs(state: AgentState) -> AgentState:
    messages = state["messages"]
    last_user_message = next(
        msg["content"] for msg in reversed(messages) if msg["role"] == "user"
    )

    try:
        retrieval_results = retriever.invoke(last_user_message)

        # Assess document relevance
        relevance_prompt = PromptTemplate(
            template="""You are a grader assessing relevance of retrieved documents to a user query.
            Here is the retrieved document to generate SQL query: \n\n {context} \n\n
            Here is the user question: {question} \n
            If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant.
            Give a binary score 'yes' or 'no' to indicate whether the document is relevant to the question.""",
            input_variables=["context", "question"],
        )

        # Combine all document content for assessment
        context = "\n\n".join(doc.page_content for doc in retrieval_results)

        class GradeSchema(BaseModel):
            binary_score: str = Field(description="Relevance score 'yes' or 'no'")

        llm_with_schema = model.with_structured_output(GradeSchema)
        chain = relevance_prompt | llm_with_schema

        grading_result = chain.invoke(
            {"question": last_user_message, "context": context}
        )
        score = grading_result.binary_score

        # Add results as system message
        system_message = {"role": "system", "content": context}

        next_step = "generate_response" if score == "yes" else "rewrite_query"

        return {"messages": messages + [system_message], "next": next_step}
    except Exception as e:
        error_message = {
            "role": "system",
            "content": f"Error retrieving documents: {str(e)}",
        }
        return {"messages": messages + [error_message], "next": "generate_response"}


def rewrite_query(state: AgentState) -> AgentState:
    messages = state["messages"]

    # Find the last user message
    last_user_message = next(
        msg["content"] for msg in reversed(messages) if msg["role"] == "user"
    )

    rewrite_prompt = f"""
    Look at the input and try to reason if the query is enough to generate a SQL query.
    Here is the initial user prompt:
    ------- 
    {last_user_message}
    -------
    Formulate an improved question that would help retrieve more relevant information for generating a SQL query.
    """

    # response = model.invoke([HumanMessage(content=rewrite_prompt)])
    response = model.invoke([{"role": "user", "content": rewrite_prompt}])
    rewritten_query = response.content

    system_message = {
        "role": "system",
        "content": f"I've rewritten your query to: {rewritten_query}",
    }

    # Add a new user message with the rewritten query
    new_user_message = {"role": "user", "content": rewritten_query}

    return {
        "messages": messages + [system_message, new_user_message],
        "next": "retrieve",
    }


def generate_sql(state: AgentState) -> AgentState:
    messages = state["messages"]

    # Extract user question and context
    last_user_message = next(
        msg["content"] for msg in reversed(messages) if msg["role"] == "user"
    )

    # Find the system message with retrieved context
    context = ""
    for msg in reversed(messages):
        if (
            msg["role"] == "system"
            and msg["content"]
            and not msg["content"].startswith("I've rewritten")
            and not msg["content"].startswith("Error")
        ):
            context = msg["content"]
            break

    prompt = PromptTemplate.from_template(
        "You are an assistant for text to SQL task. This SQL query will be passed to the database to fetch result so give me the best one.\n\n"
        "This is the question asked by the user: {question}\n\n"
        "Use the following pieces of retrieved context to generate query:\n\n{context}\n\n"
        "Return ONLY the SQL query without any explanations, wrapped in ```sql ``` code blocks."
    )

    output_parser = StrOutputParser()
    sql_chain = prompt | model | output_parser

    try:
        sql_response = sql_chain.invoke(
            {"context": context, "question": last_user_message}
        )

        # Extract SQL if in code blocks
        sql_query = extract_sql(sql_response) or sql_response

        ai_message = {"role": "assistant", "content": f"```sql\n{sql_query}\n```"}

        return {"messages": messages + [ai_message]}
    except Exception as e:
        error_message = {
            "role": "assistant",
            "content": f"I couldn't generate a SQL query. Error: {str(e)}",
        }
        return {"messages": messages + [error_message]}


def extract_sql(text):
    match = re.search(r"```sql\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else text


workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("route", route)  # Entry node
workflow.add_node("retrieve", retrieve_docs)
workflow.add_node("rewrite_query", rewrite_query)
workflow.add_node("generate_response", generate_sql)

# Set up routing logic
workflow.add_conditional_edges("route", route)

# Transitions from each node
workflow.add_edge("retrieve", "route")
workflow.add_edge("rewrite_query", "route")
workflow.add_edge("generate_response", END)

# Set the entrypoint
workflow.add_edge(START, "route")

# Compile the graph
graph = workflow.compile()


def process_message(user_message):
    inputs = {"messages": [{"role": "user", "content": user_message}]}

    # Execute and track the last assistant message
    final_state = None
    for state in graph.stream(inputs):
        final_state = state
        print(f"State update: {list(state.keys())}")

    if final_state:
        messages = final_state.get("generate_response", {}).get("messages", [])
        for msg in reversed(messages):
            if msg["role"] == "assistant":
                return msg["content"]

    return "No response generated"


# Test function
result = process_message("which product has the best reviews")
print(f"Result: {result}")

# # Create a Gradio interface
# iface = gr.Interface(
#     fn=process_message,
#     inputs="text",
#     outputs="text",
#     title="Agentic RAG with LangGraph",
#     description="Enter a question to generate an SQL query",
# )

# # Launch the Gradio app
# if __name__ == "__main__":
#     iface.launch(debug=True)
