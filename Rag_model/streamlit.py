import streamlit as st
import requests
import os

API_URL = "http://localhost:8000"

# Set layout to wide for a more professional dashboard look
st.set_page_config(page_title="LangGraph RAG Agent", layout="wide", page_icon="🛡️")

# --- SIDEBAR: Status & Architecture ---
with st.sidebar:
    st.header("⚙️ System Configuration")

    try:
        response = requests.get(f"{API_URL}/check-database")
        db_status = response.json()

        if db_status.get("exists"):
            st.success("Vector Database: Ready ✅")
            st.markdown("Multi-agent evaluation loops are **active**.")
        else:
            st.warning("Vector Database: Empty ⚠️")
    except Exception:
        st.error("Backend Server: Offline ❌")
        db_status = {"exists": False}

    st.divider()

    # Display the LangGraph Architecture Diagram
    st.header("📊 Agent Architecture")
    st.markdown("Visual representation of the iterative retrieval and generation loops.")
    st.image("rag image.png", use_container_width=True)

# --- MAIN PAGE ---
st.title("🛡️ Agentic Policy Assistant")

if not db_status.get("exists"):
    st.info("Please upload policy documents to initialize the knowledge base.")
    uploaded_file = st.file_uploader("Upload Policy PDFs", accept_multiple_files=True)

    if st.button("Upload & Ingest", type="primary"):
        if uploaded_file:
            with st.spinner("Processing, splitting, and embedding documents (MMR enabled)..."):
                files = [("files", (f.name, f, "application/pdf")) for f in uploaded_file]
                res = requests.post(f"{API_URL}/ingest", files=files)
                if res.status_code == 200:
                    st.success("Documents successfully ingested into FAISS!")
                    st.rerun()
                else:
                    st.error("Upload failed.")
else:
    question = st.text_input("Ask a specific question about company policies:",
                             placeholder="e.g., How many leave days can I bank?")

    if question and st.button("Search"):
        # st.status provides an expandable, animated loading block
        with st.status("Agent traversing the graph...", expanded=True) as status:
            response = requests.post(f"{API_URL}/ask", json={"question": question})

            if response.status_code == 200:
                data = response.json()

                # Print the exact steps the LangGraph agent took
                st.write("**Agent Internal Thought Process:**")
                for step in data.get("agent_trace", []):
                    st.code(step, language="markdown")

                status.update(label="Graph Execution Complete!", state="complete", expanded=False)

                st.subheader("Final Answer")
                st.info(data["answer"])

                with st.expander("📚 View Retrieved Context (Sources)"):
                    for i, source in enumerate(data.get("sources", [])):
                        st.markdown(f"**Source #{i + 1}:**")
                        st.markdown(f"> {source}")
            else:
                status.update(label="Execution Failed.", state="error")
                st.error("Unable to connect to the agent.")