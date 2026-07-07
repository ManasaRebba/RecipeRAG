import os
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DB_PATH = "vector_db"

def load_vector_db():

    embedding_function = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


    db = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embedding_function
    )

    return db

def retrieve_recipe(query, db):

    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k":8,
            "fetch_k":20
        }
    )

    results = retriever.invoke(query)

    if len(results) == 0:
        return []

    return results

def load_llm():
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0
    )

    return llm

def generate_answer(query, docs, llm):

    context = ""

    seen = set()

    for doc in docs:
        text = doc.page_content.strip()

        if text not in seen:
            context += "\n\n" + text
            seen.add(text)

    prompt = f"""
You are an expert Recipe Recommendation Assistant.

Answer ONLY from the retrieved recipe context.

If the recipe exists, provide:

Recipe Name

Ingredients

Instructions

Preparation Tips (if available)

Serving Size (if available)

If some information is missing, write:

Not available in the recipe collection.

If the requested recipe is NOT found in the retrieved context, reply ONLY:

Recipe not found in the recipe collection.

Never invent recipes.
Never add ingredients.
Never use outside knowledge.

Retrieved Recipe Context:

{context}

User Question:

{query}
"""

    response = llm.invoke(prompt)

    return response.content

def ask_recipe(query):
    db = load_vector_db()
    llm = load_llm()

    results = retrieve_recipe(query, db)

    answer = generate_answer(query, results, llm)

    return answer, len(results)



if __name__ == "__main__":
    query = input("Ask your recipe question: ")

    answer, docs = ask_recipe(query)

    print("\nAI Response:\n")
    print(answer)
    print(f"\nRetrieved Documents: {docs}")