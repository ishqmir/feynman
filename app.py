import os
import tempfile
import streamlit as st
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader

st.set_page_config(page_title="Feynman")
st.title("Feynman")

st.markdown("""
<style>
[data-testid="stChatMessageAvatarUser"] {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_models():
    llm = ChatOllama(model="qwen2.5:1.5b", temperature=0.2)
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    return llm, embeddings

llm, embeddings = get_models()
persist_dir = "./chroma_db"

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    if st.session_state.get("processed_file") != uploaded_file.name:
        with st.spinner("Processing PDF..."):
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_path = tmp_file.name

                loader = PyPDFLoader(tmp_path)
                documents = loader.load()
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = splitter.split_documents(documents)

                vs = Chroma(
                    collection_name="feynman_docs",
                    persist_directory=persist_dir,
                    embedding_function=embeddings
                )
                try:
                    vs.delete_collection()
                except Exception:
                    pass

                Chroma.from_documents(
                    documents=chunks,
                    embedding=embeddings,
                    persist_directory=persist_dir,
                    collection_name="feynman_docs"
                )
                st.session_state["processed_file"] = uploaded_file.name
                st.session_state["messages"] = []
                st.success("PDF indexed successfully!")
            except Exception as e:
                st.session_state["processed_file"] = None
                st.error(f"Failed to index PDF: {e}")

if st.session_state.get("processed_file"):
    try:
        vectorstore = Chroma(
            collection_name="feynman_docs",
            persist_directory=persist_dir,
            embedding_function=embeddings
        )
        if vectorstore._collection.count() == 0:
            st.warning("The vector store is empty. Please re-upload your PDF.")
            st.session_state["processed_file"] = None
        else:
            retriever = vectorstore.as_retriever(
                search_kwargs={"k": 4}
            )

            prompt = ChatPromptTemplate.from_messages([
                (
                    "system",
                    "You are a helpful research assistant named Feynman developed by Ishfaq Mir. Answer the user's question clearly, directly, and comprehensively based on the provided context. If asked for an overview or summary of the paper/document, summarize the main topics, findings, and key points from the context. Do not wrap regular text, numbers, or currency amounts in backticks or code formatting. If the answer is truly not present in the context, say 'I don't have enough information from the document to answer that'."
                ),
                (
                    "human",
                    "Context:\n{context}\n\nQuestion:\n{question}"
                ),
            ])

            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)

            rag_chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )

            if "messages" not in st.session_state:
                st.session_state["messages"] = []

            for msg in st.session_state["messages"]:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])

            if user_prompt := st.chat_input("Ask a question about the PDF..."):
                st.chat_message("user").write(user_prompt)
                st.session_state["messages"].append({"role": "user", "content": user_prompt})

                with st.chat_message("assistant"):
                    response = st.write_stream(rag_chain.stream(user_prompt))

                st.session_state["messages"].append({"role": "assistant", "content": response})
    except Exception as e:
        st.error(f"Error accessing vector database: {e}. Please re-upload your PDF.")
        st.session_state["processed_file"] = None
else:
    st.info("Upload a PDF document above to start asking questions.")