import os
import time
import json
import subprocess
import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000"

# --- Backend Auto-Spawner ---
def ensure_backend_running():
    """Ensure the FastAPI backend is active before rendering the UI."""
    try:
        res = requests.get(f"{API_URL}/", timeout=2)
        if res.status_code == 200:
            return True
    except Exception:
        pass
    
    try:
        subprocess.Popen([
            "python", "-m", "uvicorn", "backend.main:app", 
            "--host", "0.0.0.0", "--port", "8000"
        ])
        time.sleep(3)  # Allow time for server initialization
    except Exception as e:
        print(f"Failed to start backend process: {e}")

# Run startup check on boot
ensure_backend_running()

# 1. Page Configuration
st.set_page_config(
    page_title="Transformer RAG Expert", 
    page_icon="🔬", 
    layout="wide"
)

# 2. Custom CSS
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stChatMessage { border-radius: 10px; margin-bottom: 10px; }
    .st-emotion-cache-1c7n2ka { max-width: 95%; } /* Center content better */
    .sidebar-text { font-size: 14px; color: #555; }
    </style>
    """, unsafe_allow_html=True)

# 3. Sidebar - Dashboard & Management
with st.sidebar:
    st.title("📊 Control Center")
    
    # System Status Indicator
    try:
        health_check = requests.get(f"{API_URL}/")
        if health_check.status_code == 200:
            st.success("API Status: Online")
    except Exception:
        st.error("API Status: Offline (Check backend)")

    st.divider()
    
    # Metrics Section
    def load_metrics():
        try:
            with open("metrics.json", "r") as f:
                return json.load(f)
        except Exception:
            return {"faithfulness": 0, "relevancy": 0, "last_run": "N/A"}

    metrics = load_metrics()

    st.subheader("Performance Metrics")
    col1, col2 = st.columns(2)
    col1.metric("Faithfulness", f"{metrics['faithfulness']*100:.0f}%")
    col2.metric("Relevancy", f"{metrics['relevancy']*100:.0f}%")

    st.caption(f"Last benchmark run: {metrics['last_run']}")

    st.divider()
    
    # Run Benchmark Button
    st.subheader("🧪 Developer Tools")
    if st.button("🚀 Run System Benchmark", use_container_width=True):
        with st.status("Running RAGAS Evaluation...", expanded=False) as status:
            try:
                response = requests.post(f"{API_URL}/run-benchmark")
                if response.status_code == 200:
                    status.update(label="Benchmark Complete!", state="complete")
                    st.toast("Metrics updated successfully!", icon="✅")
                    time.sleep(2)
                    st.rerun()
                elif response.status_code == 422:
                    status.update(label="Validation Error", state="error")
                    st.error(f"Validation error: {response.json()}")
                else:
                    status.update(label="Benchmark Failed", state="error")
                    st.error("Check backend logs for details.")
            except Exception as e:
                status.update(label="Connection Error", state="error")
                st.error(f"Connection Error: {e}")

    st.divider()

    # Knowledge Management (Upload Endpoint)
    st.subheader("📥 Knowledge Management")
    st.markdown('<p class="sidebar-text">Add new research papers to the Vector Store.</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload PDF, TXT, or MD", type=["pdf", "txt", "md"])
    
    if uploaded_file:
        if st.button("Process & Index Document", use_container_width=True):
            with st.spinner("Brain in progress... indexing..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                try:
                    response = requests.post(f"{API_URL}/upload", files=files)
                    if response.status_code == 200:
                        st.balloons()
                        st.success(f"Indexed: {uploaded_file.name}")
                    else:
                        st.error(f"Error: {response.json().get('detail')}")
                except Exception as e:
                    st.error(f"Connection Error: {e}")

    st.divider()
    if st.button("Clear Chat History", type="secondary", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# 4. Main Chat UI Header
st.title("🔬 Research Assistant: Transformer Architecture")
st.info("Ask complex questions about Multi-Head Attention, Positional Encodings, or any uploaded papers.")

# 5. Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("🔍 View Evidence (Retrieved Context)"):
                for idx, src in enumerate(message["sources"]):
                    st.caption(f"Source Chunk {idx+1}:")
                    st.write(src)
                    if idx < len(message["sources"]) - 1:
                        st.divider()

# 6. Chat Input & Generation Logic
if prompt := st.chat_input("What would you like to know?"):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process Assistant Response
    with st.chat_message("assistant"):
        api_success = False
        answer = ""
        sources = []
        
        # Phase 1: Call API inside status container
        with st.status("Consulting the knowledge base...", expanded=True) as status:
            try:
                payload = {"prompt": prompt}
                response = requests.post(f"{API_URL}/ask", json=payload)
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("answer", "")
                    sources = data.get("sources", [])
                    api_success = True
                    status.update(label="Analysis Complete!", state="complete", expanded=False)
                else:
                    status.update(label="API Error", state="error", expanded=False)
                    st.error("The backend returned an error. Please check your API logs.")
            except Exception as e:
                status.update(label="Connection Failed", state="error", expanded=False)
                st.error(f"Could not connect to the API: {e}")

        # Phase 2: Render results OUTSIDE status container to avoid nested expander errors
        if api_success:
            st.markdown(answer)
            
            if sources:
                with st.popover("🔍 View Evidence (Retrieved Context)"):
                    for idx, src in enumerate(sources):
                        st.caption(f"Source Chunk {idx+1}:")
                        st.write(src)
                        if idx < len(sources) - 1:
                            st.divider()

            # Append to session state
            st.session_state.messages.append({
                "role": "assistant", 
                "content": answer, 
                "sources": sources
            })