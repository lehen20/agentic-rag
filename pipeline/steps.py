from langchain_core.messages import HumanMessage
from prompts.templates import grade_prompt, rewrite_prompt
from models.ollama_model import get_llm
from langchain_core.output_parsers import StrOutputParser
from langchain import hub
from pydantic import BaseModel, Field

def grade_documents(state):
    
    class grade(BaseModel):
        binary_score: str = Field(description="Relevance score 'yes' or 'no'")
        
    llm_with_tool = get_llm().with_structured_output(grade)
    prompt = grade_prompt
    chain = prompt | llm_with_tool 
    
    messages = state["messages"]
    question = messages[0].content
    docs = messages[-1].content
    print(f"Grade {question=}, {docs=}")
    print(f"{chain=}")
    result = chain.invoke({"question": question, "context": docs})
    print(f"{result.binary_score=}")
    return "generate" if result.binary_score == "yes" else "rewrite"

def rewrite(state):
    print(f"Rewrite {state=}")
    question = state["messages"][0].content
    response = get_llm().invoke([HumanMessage(content=rewrite_prompt.format(question=question))])
    print(f"Rewrite {response=}")
    return {"messages": [response]}

def generate(state):
    question = state["messages"][0].content
    docs = state["messages"][-1].content
    prompt = hub.pull("rlm/rag-prompt")
    rag_chain = prompt | get_llm() | StrOutputParser()
    print(f"{rag_chain=}")
    response = rag_chain.invoke({"context": docs, "question": question})
    return {"messages": [response]}
