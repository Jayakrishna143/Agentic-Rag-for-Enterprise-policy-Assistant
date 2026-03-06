import os
import shutil
import asyncio
from typing import List, TypedDict, Annotated
import operator
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, StateGraph, START

app = FastAPI()
DB_PATH = r"C:\Users\jayak\OneDrive\Desktop\fastapi\Rag_model\vectordb"
TEMP_UPLOAD_PATH = r"C:\Users\jayak\OneDrive\Desktop\fastapi\Rag_model\temp_uploads"
os.makedirs(TEMP_UPLOAD_PATH, exist_ok=True)

class Question(BaseModel):
    question: str

# --- GRAPH STATE ---
class GraphState(TypedDict):
    question: str
    context: str
    documents: List[str]
    generation: str
    feedback: str
    trace: Annotated[List[str], operator.add] # Appends to the list instead of overwriting
    retries: int

# --- PROMPTS ---
EVAL_CONTEXT_PROMPT = PromptTemplate.from_template("""
Respond with exactly 'PASS' or 'FAIL' on the first line. On the next line, briefly explain why.
Does this Context contain enough relevant information to answer the Question?
Question: {question}
Context: {context}
""")

REWRITE_PROMPT = PromptTemplate.from_template("""
The previous search query failed. Rewrite the question into a better search query. Focus on keywords.
Original Question: {question}
Feedback on failure: {feedback}
Output ONLY the rewritten query.
""")

EVAL_UTILIZATION_PROMPT = PromptTemplate.from_template("""
Respond with exactly 'PASS' or 'FAIL' on the first line. 
Did the Generated Answer use all the important information from the Context, or did it omit key details?
Question: {question}
Context: {context}
Answer: {answer}
""")

EVAL_RELEVANCE_PROMPT = PromptTemplate.from_template("""
Respond with exactly 'PASS' or 'FAIL' on the first line. 
Does the Answer directly, accurately, and concisely address the Question?
Question: {question}
Answer: {answer}
""")

GENERATION_PROMPT = PromptTemplate.from_template("""
Answer based ONLY on the context below. If you don't know, say "I don't know based on the documents."
{feedback_section}
Context: {context}
Question: {question}
Answer:
""")

# --- GLOBAL LLM INIT ---
llm = OllamaLLM(model="gemma3:1b", temperature=0)

# --- GRAPH NODES ---
def retrieve_node(state: GraphState):
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    db = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_type = "mmr",search_kwargs={"k": 3,"fetch_k":20,"lambda_mult":0.5})
    
    docs = retriever.invoke(state["question"])
    context = "\n\n".join([doc.page_content for doc in docs])
    doc_sources = [doc.page_content for doc in docs]
    
    return {"context": context, "documents": doc_sources, "trace": [f" Retrieved context for: '{state['question']}'"]}

async def eval_context_node(state: GraphState):
    chain = EVAL_CONTEXT_PROMPT | llm | StrOutputParser()
    eval_result = await chain.ainvoke({"question": state["question"], "context": state["context"]})
    
    if "PASS" in eval_result.upper().split('\n')[0]:
        return {"feedback": "PASS", "trace": [" Context Precision: PASS"]}
    else:
        return {"feedback": eval_result, "trace": [f" Context Precision: FAIL. Reason: {eval_result.split(maxsplit=1)[-1]}"]}

async def rewrite_node(state: GraphState):
    chain = REWRITE_PROMPT | llm | StrOutputParser()
    new_query = await chain.ainvoke({"question": state["question"], "feedback": state["feedback"]})
    return {"question": new_query.strip(), "retries": state["retries"] + 1, "trace": [f" Rewrote query to: '{new_query.strip()}'"]}

async def generate_node(state: GraphState):
    feedback_section = f"PREVIOUS FEEDBACK TO FIX:\n{state['feedback']}" if state["feedback"] and state["feedback"] != "PASS" else ""
    chain = GENERATION_PROMPT | llm | StrOutputParser()
    answer = await chain.ainvoke({"context": state["context"], "question": state["question"], "feedback_section": feedback_section})
    return {"generation": answer, "trace": [f" Generated answer attempt {state['retries'] + 1}"]}

async def eval_generation_node(state: GraphState):
    util_chain = EVAL_UTILIZATION_PROMPT | llm | StrOutputParser()
    rel_chain = EVAL_RELEVANCE_PROMPT | llm | StrOutputParser()

    # Run parallel evaluations
    util_eval, rel_eval = await asyncio.gather(
        util_chain.ainvoke({"question": state["question"], "context": state["context"], "answer": state["generation"]}),
        rel_chain.ainvoke({"question": state["question"], "answer": state["generation"]})
    )

    util_pass = "PASS" in util_eval.upper().split('\n')[0]
    rel_pass = "PASS" in rel_eval.upper().split('\n')[0]

    feedback = ""
    trace_updates = []
    
    if util_pass and rel_pass:
        return {"feedback": "PASS", "trace": [" Info Utilization: PASS", " Answer Relevance: PASS"]}
    
    if not util_pass:
        trace_updates.append(" Info Utilization: FAIL")
        feedback += f"You missed key details. Include: {util_eval}\n"
    if not rel_pass:
        trace_updates.append(" Answer Relevance: FAIL")
        feedback += f"Make the answer more direct: {rel_eval}\n"
        
    return {"feedback": feedback, "retries": state["retries"] + 1, "trace": trace_updates}

# --- CONDITIONAL EDGES ---
def route_after_context(state: GraphState):
    if state["feedback"] == "PASS":
        return "generate"
    elif state["retries"] >= 2:
        return "generate" # Force generation if max retries hit
    return "rewrite"

def route_after_generation(state: GraphState):
    if state["feedback"] == "PASS" or state["retries"] >= 2:
        return END
    return "generate"

# --- BUILD GRAPH ---
workflow = StateGraph(GraphState)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("eval_context", eval_context_node)
workflow.add_node("rewrite", rewrite_node)
workflow.add_node("generate", generate_node)
workflow.add_node("eval_generation", eval_generation_node)

workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "eval_context")
workflow.add_conditional_edges("eval_context", route_after_context, {"generate": "generate", "rewrite": "rewrite"})
workflow.add_edge("rewrite", "retrieve")
workflow.add_edge("generate", "eval_generation")
workflow.add_conditional_edges("eval_generation", route_after_generation, {END: END, "generate": "generate"})

app_graph = workflow.compile()

# --- API ENDPOINTS ---
@app.get("/check-database")
def check_database():
    exists = os.path.exists(DB_PATH) and len(os.listdir(DB_PATH)) > 0
    return {"exists": exists}

@app.post("/ingest")
async def ingest_pdfs(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")
    try:
        documents = []
        for file in files:
            file_path = os.path.join(TEMP_UPLOAD_PATH, file.filename)
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)

        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        db = FAISS.from_documents(chunks, embeddings)

        os.makedirs(DB_PATH, exist_ok=True)
        db.save_local(DB_PATH)
        
        shutil.rmtree(TEMP_UPLOAD_PATH)
        os.makedirs(TEMP_UPLOAD_PATH, exist_ok=True)

        return {"message": "Database created successfully", "chunks": len(chunks)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask")
async def ask_question(q: Question):
    if not os.path.exists(DB_PATH):
        raise HTTPException(status_code=404, detail="Database not found")

    try:
        initial_state = {
            "question": q.question, 
            "context": "", 
            "documents": [], 
            "generation": "", 
            "feedback": "", 
            "trace": [" Graph execution started..."],
            "retries": 0
        }
        
        # Run the graph
        final_state = await app_graph.ainvoke(initial_state)

        return {
            "answer": final_state["generation"],
            "sources": final_state["documents"],
            "agent_trace": final_state["trace"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)