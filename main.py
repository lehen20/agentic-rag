from langchain_community.vectorstores import LanceDB
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langgraph.prebuilt import ToolNode, tools_condition
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, StateGraph
from langchain_core.documents import Document
from langchain.agents import initialize_agent, AgentType
from langchain_core.tools import Tool
from langchain.agents import AgentExecutor


import gradio as gr
from typing import Annotated, Literal, Sequence, TypedDict
from pydantic import BaseModel, Field
import lancedb
import re

from src.redis import get_redis
from src.model import model


get_redis()

model = model

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
embeddings = embeddings_mini

db = lancedb.connect("/tmp/lancedb")

vectorstore = LanceDB.from_documents(
    documents=documents,
    embedding=embeddings,
)
retriever = vectorstore.as_retriever()


retriever_tool = Tool(
    name="get_sql_query_from_pdf",
    description="Generate an SQL query based on the user's question and database schema",
    func=lambda input: retriever.invoke(input),  # input is a string (user query)
)


# retriever_tool = create_retriever_tool(
#     retriever,
#     "get_sql_query_from_pdf",
#     "Generate an sql query depending on the question asked by the user taking into account the database schema",
# )

tools = [retriever_tool]
print(f"{tools=}")

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


def grade_documents(state) -> Literal["generate", "rewrite"]:
    class grade(BaseModel):
        binary_score: str = Field(description="Relevance score 'yes' or 'no'")

    llm_with_tool = model.with_structured_output(grade)
    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved subset of document to a user query. \n
        Here is the retrieved document to generate sql query: \n\n {context} \n\n
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
    
    print(f"{score=}")

    return "generate" if score == "yes" else "rewrite"



def agent(state):
    messages = state["messages"]
    question = messages[-1].content
    print(f"{question=}")
    
    llm = ChatOllama(model="llama3.1", temperature=0)

    agent_executor = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        return_intermediate_steps=True, # <-- crucial
        handle_parsing_errors= True
    )

    try:
        result = agent_executor.invoke({"input": question})

        # If there's an intermediate step (tool usage), return only the first one
        if "intermediate_steps" in result and result["intermediate_steps"]:
            first_obs = result["intermediate_steps"][0]
            observation = first_obs[1]  # (ActionLog, Observation)
            return {"messages": [observation]}

        # Otherwise return whatever final answer the model gave
        return {"messages": [result["output"]]}
    
    except Exception as e:
        print(f"Agent execution error: {e}")
        return {"messages": [f"Error in processing: {str(e)}"]}

    # model = ChatOllama(model="llama3.1", temperature=0)
    # model = model.bind_tools(tools)
    # response = model.invoke(messages)


def rewrite(state):
    messages = state["messages"]
    question = messages[0].content
    msg = [
        HumanMessage(
            content=f""" \n
            Look at the input and try to reason if the query is enough to generate a sql query. \n
            Here is the initial user prompt:
            \n ------- \n
            {question}
            \n ------- \n
            Formulate an improved question, only if required """,
        )
    ]
    response = model.invoke(msg)
    return {"messages": [response]}



def generate(state):
    messages = state["messages"]
    question = messages[0].content
    last_message = messages[-1]
    
    # Better handling of document extraction
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        # If it's a tool response with documents
        docs = []
        for call in last_message.tool_calls:
            if call.get("documents"):
                docs.extend(call["documents"])
    elif isinstance(last_message.content, list) and all(isinstance(doc, Document) for doc in last_message.content):
        docs = last_message.content
    elif isinstance(last_message.content, str):
        # Try to parse content as document
        content = last_message.content
        docs = [Document(page_content=content)]
    else:
        print(f"Unexpected message format: {type(last_message.content)}")
        return {"messages": ["Unable to process documents from the message."]}
    
    # Format documents for context
    context = "\n\n".join(doc.page_content for doc in docs)
    
    prompt = PromptTemplate.from_template(
        "You are an assistant for text to SQL task. This SQL query will be passed to the database to fetch result so give me the best one.\n\n"
        "This is the question asked by the user: {question}\n\n"
        "Use the following pieces of retrieved context to generate query:\n\n{context}"
    )
    
    output_parser = StrOutputParser()
    rag_chain = prompt | model | output_parser
    
    print(f"Retrieved context:\n{context}")
    response = rag_chain.invoke({"context": context, "question": question})
    return {"messages": [response]}

workflow = StateGraph(AgentState)
workflow.add_node("agent", agent)
retrieve = ToolNode([retriever_tool])
print(f"{retrieve=}")
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
    print(f"{inputs=}")
    for output in graph.stream(inputs):
        print(f"Debug output: {output}") 

        # Check if we're at the final generate step
        if "generate" in output and "messages" in output["generate"]:
            messages = output["generate"]["messages"]
            if messages:
                content_output = messages[0] if isinstance(messages[0], str) else messages[0].content
                print(f"Extracted content from generate: {content_output}")
                return content_output

        # Alternatively handle other message formats, like agent/tool output
        if "agent" in output and "messages" in output["agent"]:
            messages = output["agent"]["messages"]
            for msg in messages:
                if hasattr(msg, "tool_calls"):
                    for call in msg.tool_calls:
                        if "args" in call and "query" in call["args"]:
                            content_output = call["args"]["query"]
                            print(f"Extracted SQL query: {content_output}")
                            return content_output

    return content_output or "No relevant info found"

def extract_sql(text):
    match = re.search(r"```sql\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else None

process_message("which product has the best reviews")

# # Create a Gradio interface
# iface = gr.Interface(
#     fn=process_message,
#     inputs="text",
#     outputs="text",
#     title="Agentic RAG",
#     description="Enter a message to query",
# )
# # Launch the Gradio app
# iface.launch(debug=True)
