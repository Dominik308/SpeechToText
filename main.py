"""Einfacher Konsolen-Chat mit den eigenen Dokumenten (RAG) – ohne GUI."""
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

model = OllamaLLM(model="mirage335/Llama-3_1-8B-Instruct-abliterated-virtuoso")

template = """
Du bist ein hilfsbereiter Assistent. Antworte immer auf Deutsch.
Nutze die folgenden Auszüge aus den Dokumenten, wenn sie zur Frage passen.

Dokumente: {context}

Frage: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

while True:
    print("\n\n-------------------------------")
    question = input("Deine Frage (q zum Beenden): ")
    print("\n\n")
    if question == "q":
        break

    context = retriever.invoke(question)
    result = chain.invoke({"context": context, "question": question})
    print(result)
