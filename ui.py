import streamlit as st
from app import ask_recipe

st.set_page_config(
    page_title="Recipe Recommendation Agent",
    layout="wide"
)

# ---------------- Sidebar ----------------

st.sidebar.title("Recipe Agent")

st.sidebar.markdown("---")

st.sidebar.header("Example Questions")

st.sidebar.markdown("""
- Stuffed Capsicum
- Tomato Soup
- Vegetable Biryani
- Green Manchurian
- Suggest recipes using potatoes
""")

st.sidebar.markdown("---")

st.sidebar.header("About")

st.sidebar.info("""
This application uses:

-  Recipe PDFs
-  Groq Llama 3.3
-  ChromaDB
-  LangChain
-  Streamlit
""")

if "messages" not in st.session_state:
    st.session_state.messages = []

if st.sidebar.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()


# ---------------- Main Page ----------------

st.title(" Recipe Recommendation Agent")

st.caption(
    "Discover delicious recipes using AI-powered semantic search and Retrieval-Augmented Generation (RAG)."
)

st.markdown("---")



# Display previous messages

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ---------------- Chat Input ----------------

query = st.chat_input("Ask me about any recipe...")

if query:

    # Show user message

    st.session_state.messages.append(
        {"role": "user", "content": query}
    )

    with st.chat_message("user"):
        st.markdown(query)

    # Generate answer
    try:
        with st.spinner("Finding the best recipe for you..."):
            answer, docs = ask_recipe(query)
            st.success("Recipe Retrieved Successfully!")
            st.caption(f"Retrieved {docs} relevant recipe document(s)")

    except Exception:
        st.error("Unable to connect to the AI model. Please try again.")
        st.stop()

    # Show assistant message

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )

    with st.chat_message("assistant"):
        st.markdown(answer)
