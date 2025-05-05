from langchain_community.vectorstores import LanceDB
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langgraph.graph.message import add_messages
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, StateGraph
from langchain.tools.retriever import create_retriever_tool
from langchain_core.messages import BaseMessage
from langchain import hub
import gradio as gr
from typing import Annotated, Literal, Sequence, TypedDict
from pydantic import BaseModel, Field
import lancedb
from src.redis import get_redis
from src.model import model
from lancedb.pydantic import Vector, LanceModel
import pyarrow as pa

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


text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=100, chunk_overlap=50
)
doc_splits = text_splitter.split_documents(docs)

schema = pa.schema([
    pa.field("id", pa.string()),
    pa.field("text", pa.string()),
    
    # Metadata fields flattened
    pa.field("metadata", pa.struct([
        pa.field("producer", pa.string(), True),
        pa.field("creator", pa.string(), True),
        pa.field("creationdate", pa.string(), True),
        pa.field("source", pa.string(), True),
        pa.field("total_pages", pa.int32(), True),
        pa.field("page", pa.int32(), True),
        pa.field("page_label", pa.string(), True),
    ])),
    
    # Vector field: 384-dimensional float32
    pa.field("vector", pa.list_(pa.float32(), 384)),
])


# for doc in doc_splits:

#     # Initialize the metadata for each document (if necessary)
#     doc.metadata = {
#         "producer": "ProducerName",
#         "creator": "CreatorName",
#         "creationdate": "2025-05-04",
#         "source": "source_name_or_pdf_path",
#         "total_pages": 10,
#         "page": 1,
#         "page_label": "Page 1"
#     }

db = lancedb.connect("my_lancedb")

# # Use LangChain or custom LanceDB wrapper to create the table
table = db.create_table("docs", schema=schema, mode="overwrite") 

# Add to lancedb as vectordb
vectorstore = LanceDB.from_documents(
    connection=db,
    table_name="docs",
    documents=doc_splits,
    embedding=embeddings_mini,
)
retriever = vectorstore.as_retriever()


# create the tools
retriever_tool = create_retriever_tool(
    retriever,
    "get_sql_query_from_pdf",
    "Generate an SQL query based on the user's question and database schema",
)

tools = [retriever_tool]
retrieve = ToolNode(tools)


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


def grade_documents(state) -> Literal["generate", "rewrite"]:
    class grade(BaseModel):
        binary_score: str = Field(description="Relevance score 'yes' or 'no'")

    model = ChatOllama(model="llama3.1", temperature=0)

    llm_with_tool = model.with_structured_output(grade)
    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question. \n
        Here is the retrieved document: \n\n {context} \n\n
        Here is the user question: {question} \n
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
        input_variables=["context", "question"],
    )
    chain = prompt | llm_with_tool

    messages = state["messages"]
    last_message = messages[-1]
    question = messages[0].content
    docs = last_message.content

    scored_result = chain.invoke({"question": question, "context": docs})
    score = scored_result.binary_score

    return "generate" if score == "yes" else "rewrite"


def agent(state):
    messages = state["messages"]
    model = ChatOllama(model="llama3.1", temperature=0)
    model = model.bind_tools(tools)
    response = model.invoke(messages)
    return {"messages": [response]}


def rewrite(state):
    messages = state["messages"]
    question = messages[0].content
    msg = [
        HumanMessage(
            content=f""" \n
            Look at the input and try to reason about the underlying semantic intent / meaning. \n
            Here is the initial question:
            \n ------- \n
            {question}
            \n ------- \n
            Formulate an improved question: """,
        )
    ]
    model = ChatOllama(model="llama3.1", temperature=0)

    response = model.invoke(msg)
    return {"messages": [response]}


def generate(state):
    messages = state["messages"]
    question = messages[0].content
    last_message = messages[-1]
    docs = last_message.content

    prompt = hub.pull("rlm/rag-prompt")
    llm= ChatOllama(model="llama3.1", temperature=0)


    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = prompt | llm | StrOutputParser()
    response = rag_chain.invoke({"context": docs, "question": question})
    return {"messages": [response]}


workflow = StateGraph(AgentState)
workflow.add_node("agent", agent)
# retrieve = ToolNode([retriever_tool])
workflow.add_node("retrieve", retrieve)
workflow.add_node("rewrite", rewrite)
workflow.add_node("generate", generate)
workflow.set_entry_point("agent")
workflow.add_conditional_edges(
    "agent", tools_condition, {"tools": "retrieve", END: END}
)
workflow.add_conditional_edges("retrieve", grade_documents)
workflow.add_edge("generate", END)
workflow.add_edge("rewrite", "agent")
graph = workflow.compile()


def process_message(user_message):
    inputs = {"messages": [("user", user_message)]}
    content_output = None
    for output in graph.stream(inputs):
        print(f"Debug output: {output}")  # Debugging line to print the output
        if "generate" in output and "messages" in output["generate"]:
            messages = output["generate"]["messages"]
            content_output = messages[0] if messages else None
            print(f"Extracted content: {content_output}") 

    return content_output if content_output else "No relevant output found."




# Create a Gradio interface
iface = gr.Interface(
    fn=process_message,
    inputs="text",
    outputs="text",
    title="Agentic RAG ",
    description="Enter a message to query",
)

# Launch the Gradio app
iface.launch(debug=True)