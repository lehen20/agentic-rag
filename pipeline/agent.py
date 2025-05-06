from models.ollama_model import get_llm

def agent(state, tools):
    print(f"Agent {state=}")
    model = get_llm().bind_tools(tools)
    response = model.invoke(state["messages"])
    print(f"Agent {response=}")
    return {"messages": [response]}
