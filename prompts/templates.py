from langchain_core.prompts import PromptTemplate

grade_prompt = PromptTemplate(
    template="""You are a grader assessing relevance of a retrieved document to a user question.
    Document:\n\n{context}\n\nQuestion:\n{question}
    Grade it as 'yes' if relevant, else 'no'.""",
    input_variables=["context", "question"]
)

rewrite_prompt = PromptTemplate(
    template="""Look at the input and try to reason about the underlying semantic intent.
    Original question: \n ------- \n {question} \n ------- \n
    Formulate an improved question:""",
    input_variables=["question"]
)
