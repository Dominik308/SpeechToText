from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

model = OllamaLLM(model="mirage335/Llama-3_1-8B-Instruct-abliterated-virtuoso")

template = """
You are a helpful assistant. Use the provided context to answer the user's question.

Context: {context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

while True:
    print("\n\n-------------------------------")
    question = input("Ask your question (q to quit): ")
    print("\n\n")
    if question == "q":
        break
    
    context = retriever.invoke(question)
    result = chain.invoke({"context": context, "question": question})
    print(result)