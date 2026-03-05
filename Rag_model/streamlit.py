import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(page_title="LangGraph RAG Agent", layout="centered")
st.title("🛡️ LangGraph Policy Agent")

try:
    response = requests.get(f"{API_URL}/check-database")
    db_status = response.json()
except Exception:
    st.error("Cannot connect to backend database.")
    st.stop()

if not db_status.get("exists"):
    st.warning("Database does not exist.")
    uploaded_file = st.file_uploader("Upload PDFs", accept_multiple_files=True)
    if st.button("Upload"):
        if uploaded_file:
            with st.spinner("Uploading & Embedding..."):
                files = [("files", (f.name, f, "application/pdf")) for f in uploaded_file]
                res = requests.post(f"{API_URL}/ingest", files=files)
                if res.status_code == 200:
                    st.success("Uploaded!")
                    st.rerun()
                else:
                    st.error("Upload failed.")
else:
    st.sidebar.success("LangGraph Node: Ready ✅")
    question = st.text_input("Ask a question about company policies:")

    if question:
        with st.status("Agent traversing the graph...", expanded=True) as status:
            response = requests.post(f"{API_URL}/ask", json={"question": question})
            
            if response.status_code == 200:
                data = response.json()
                
                # Print the LangGraph execution trace
                st.write("**Graph Trace:**")
                for step in data.get("agent_trace", []):
                    st.write(step)
                
                status.update(label="Graph Execution Complete!", state="complete", expanded=False)
                
                st.subheader("Final Answer")
                st.markdown(data["answer"])

                with st.expander("View Retrieved Context"):
                    for i, source in enumerate(data.get("sources", [])):
                        st.markdown(f"**Source #{i+1}:**")
                        st.info(source)
            else:
                status.update(label="Execution Failed.", state="error")
                st.error("Unable to answer.")