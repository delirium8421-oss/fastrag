"""
Streamlit Query Interface for Hybrid GraphRAG
Interactive chat interface for querying pre-indexed graph data.
"""

import streamlit as st
import asyncio
import os
import requests
import json
import logging
from typing import Dict, List, Optional, Tuple

# Import HybridRAG classes (same pattern as run_hybrid_rag.py)
from hybrid_graph_rag import HybridGraphRAG, HybridRAGFactory

# Configure logging
logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

# System prompt (from run_hybrid_rag.py lines 49-66)
SYSTEM_PROMPT = """
---Role---
You are a helpful assistant responding to user queries only in English.

---Goal---
Generate direct and concise one-sentence answers based strictly on the provided Knowledge Base.
Respond in plain text without explanations or formatting.
Provide your answer in only one sentence to answer the question adequately.
The sentence structure must be subject, verb and object, active voice.
Maintain conversation continuity and use the same language as the query.
If the answer is unknown, respond with "I don't know".

---Conversation History---
{history}

---Knowledge Base---
{context_data}
"""


def validate_ollama_server(ollama_url: str) -> Tuple[bool, Optional[str]]:
    """
    Validate Ollama server connectivity and check /api/tags endpoint.
    (From run_hybrid_rag.py lines 80-124)

    Args:
        ollama_url: Base URL of Ollama server (e.g., http://127.0.0.1:8500)

    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        # Ensure URL doesn't have trailing slash
        ollama_url = ollama_url.rstrip('/')

        # Test connection to /api/tags
        tags_url = f"{ollama_url}/api/tags"
        logger.info(f"🔗 Testing Ollama server at: {tags_url}")

        response = requests.get(tags_url, timeout=5)

        if response.status_code != 200:
            return False, f"Server returned status {response.status_code}"

        # Parse response
        data = response.json()

        if "models" not in data:
            return False, "Response missing 'models' field"

        models = data.get("models", [])

        if not models:
            return False, "No models available on the Ollama server"

        logger.info(f"✅ Ollama server is accessible with {len(models)} model(s)")
        return True, None

    except requests.exceptions.ConnectionError:
        return False, f"Cannot connect to Ollama server at {ollama_url}"
    except requests.exceptions.Timeout:
        return False, f"Connection timeout for Ollama server at {ollama_url}"
    except json.JSONDecodeError:
        return False, "Invalid JSON response from Ollama server"
    except Exception as e:
        return False, f"Error testing Ollama server: {str(e)}"


def get_available_models(ollama_url: str) -> Optional[List[Dict]]:
    """
    Fetch available models from Ollama server.
    (From run_hybrid_rag.py lines 127-150)

    Args:
        ollama_url: Base URL of Ollama server

    Returns:
        List of model dicts with 'name' field, or None if failed
    """
    try:
        ollama_url = ollama_url.rstrip('/')
        tags_url = f"{ollama_url}/api/tags"
        response = requests.get(tags_url, timeout=5)

        if response.status_code == 200:
            data = response.json()
            return data.get("models", [])

        return None

    except Exception as e:
        logger.error(f"❌ Failed to get models: {e}")
        return None


def run_async(coro):
    """
    Run async coroutine in Streamlit.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


async def initialize_hybrid_rag(working_dir: str, llm_model: str, embed_model: str, ollama_url: str):
    """
    Initialize HybridGraphRAG instance for querying existing graph.
    (Based on run_hybrid_rag.py lines 338-344)

    Args:
        working_dir: Directory containing graph data
        llm_model: LLM model name
        embed_model: Embedding model name
        ollama_url: Ollama server URL

    Returns:
        Initialized HybridGraphRAG instance
    """
    hybrid_rag = HybridRAGFactory.create_ollama_instance(
        working_dir=working_dir,
        llm_model=llm_model,
        embed_model=embed_model,
        llm_url=ollama_url,
        llm_model_max_async=2
    )
    return hybrid_rag


async def query_graph(hybrid_rag: HybridGraphRAG, question: str, top_k: int, query_type: str):
    """
    Execute query against loaded graph.
    (Based on run_hybrid_rag.py lines 370-374)

    Args:
        hybrid_rag: Initialized HybridGraphRAG instance
        question: User question
        top_k: Number of top documents to retrieve
        query_type: Type of query ("hybrid", "local", or "global")

    Returns:
        Tuple of (answer, context_text)
    """
    answer, context_text = await hybrid_rag.query(
        question,
        top_k=top_k,
        query_type=query_type
    )
    return answer, context_text


def main():
    """Main Streamlit application."""

    # Page configuration
    st.set_page_config(
        page_title="Hybrid GraphRAG Query",
        page_icon="🔍",
        layout="wide"
    )

    # Initialize session state
    if 'hybrid_rag' not in st.session_state:
        st.session_state.hybrid_rag = None
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'ollama_url' not in st.session_state:
        st.session_state.ollama_url = "http://172.26.208.1:8500"
    if 'models_loaded' not in st.session_state:
        st.session_state.models_loaded = False
    if 'available_models' not in st.session_state:
        st.session_state.available_models = []
    if 'graph_loaded' not in st.session_state:
        st.session_state.graph_loaded = False

    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Ollama Server Setup
        st.subheader("1. Ollama Server")
        ollama_url = st.text_input(
            "Ollama Server URL",
            value=st.session_state.ollama_url,
            help="URL of the Ollama server (e.g., http://172.26.208.1:8500)"
        )

        if st.button("🔗 Connect to Ollama", use_container_width=True):
            with st.spinner("Connecting to Ollama server..."):
                is_valid, error_msg = validate_ollama_server(ollama_url)

                if is_valid:
                    models = get_available_models(ollama_url)
                    if models:
                        st.session_state.ollama_url = ollama_url
                        st.session_state.available_models = models
                        st.session_state.models_loaded = True
                        st.success(f"✅ Connected! Found {len(models)} model(s)")
                    else:
                        st.error("❌ Failed to fetch models")
                else:
                    st.error(f"❌ Connection failed: {error_msg}")
                    st.session_state.models_loaded = False

        # Display connection status
        if st.session_state.models_loaded:
            st.info(f"✓ Connected to {st.session_state.ollama_url}")

        st.divider()

        # Model Selection
        st.subheader("2. Model Selection")

        if st.session_state.models_loaded:
            model_names = [m.get("name", "Unknown") for m in st.session_state.available_models]

            llm_model = st.selectbox(
                "LLM Model",
                options=model_names,
                help="Model for text generation"
            )

            embed_model = st.selectbox(
                "Embedding Model",
                options=model_names,
                help="Model for generating embeddings"
            )

            # Store selected models in session state
            st.session_state.llm_model = llm_model
            st.session_state.embed_model = embed_model
        else:
            st.warning("⚠️ Connect to Ollama server first")

        st.divider()

        # Graph Data Loading
        st.subheader("3. Load Graph Data")

        # File selection method
        selection_method = st.radio(
            "Select graph data by:",
            options=["Browse File System", "Enter Path Manually"],
            horizontal=True
        )

        working_dir = None

        if selection_method == "Browse File System":
            graphml_file = st.file_uploader(
                "Upload GraphML File",
                type=['graphml'],
                help="Select the .graphml file from your indexed graph data"
            )

            if graphml_file is not None:
                # Save uploaded file to temp location
                temp_dir = os.path.join(os.path.expanduser("~"), ".streamlit_graphrag_temp")
                os.makedirs(temp_dir, exist_ok=True)

                # Save the graphml file
                graphml_path = os.path.join(temp_dir, graphml_file.name)
                with open(graphml_path, "wb") as f:
                    f.write(graphml_file.getbuffer())

                # Use temp directory as working_dir
                working_dir = temp_dir
                st.info(f"📄 File uploaded: {graphml_file.name}")

                # Note about other required files
                st.warning("⚠️ Note: The working directory should also contain vector indexes and other RAG artifacts. If using file upload, you may need to upload all files from the working directory.")

        else:  # Enter Path Manually
            working_dir = st.text_input(
                "Working Directory Path",
                value="",
                help="Path to directory containing graph data (e.g., ./hybrid_rag_workspace/Medical)"
            )

        if st.button("📂 Load Graph", use_container_width=True):
            if not st.session_state.models_loaded:
                st.error("❌ Please connect to Ollama and select models first")
            elif not working_dir:
                st.error("❌ Please provide a working directory path or upload a file")
            elif selection_method == "Enter Path Manually" and not os.path.isdir(working_dir):
                st.error(f"❌ Directory not found: {working_dir}")
            else:
                with st.spinner("Loading graph data..."):
                    try:
                        hybrid_rag = run_async(
                            initialize_hybrid_rag(
                                working_dir,
                                st.session_state.llm_model,
                                st.session_state.embed_model,
                                st.session_state.ollama_url
                            )
                        )
                        st.session_state.hybrid_rag = hybrid_rag
                        st.session_state.graph_loaded = True
                        st.session_state.working_dir = working_dir
                        st.success("✅ Graph loaded successfully!")
                    except Exception as e:
                        st.error(f"❌ Failed to load graph: {str(e)}")
                        logger.exception("Graph loading error")

        # Display graph status
        if st.session_state.graph_loaded:
            st.info(f"✓ Graph loaded from: {st.session_state.working_dir}")

        st.divider()

        # Query Configuration
        st.subheader("4. Query Settings")

        top_k = st.number_input(
            "Top K Results",
            min_value=1,
            max_value=20,
            value=5,
            help="Number of top documents to retrieve"
        )

        query_type = st.selectbox(
            "Query Type",
            options=["hybrid", "local", "global"],
            index=0,
            help="Type of query to execute"
        )

        # Store query settings in session state
        st.session_state.top_k = top_k
        st.session_state.query_type = query_type

        st.divider()

        # Clear chat button
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    # Main chat interface
    st.title("🔍 Hybrid GraphRAG Query Interface")

    # Display system prompt in expander
    with st.expander("ℹ️ System Prompt"):
        st.code(SYSTEM_PROMPT, language="text")

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

            # Display context if available (for assistant messages)
            if message["role"] == "assistant" and "context" in message:
                with st.expander("📚 Retrieved Context"):
                    st.text(message["context"])

    # Chat input
    if prompt := st.chat_input("Ask a question about the graph..."):
        if st.session_state.hybrid_rag is None:
            st.error("❌ Please load a graph first!")
        else:
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Display user message immediately
            with st.chat_message("user"):
                st.write(prompt)

            # Execute query
            with st.chat_message("assistant"):
                with st.spinner("Querying graph..."):
                    try:
                        answer, context = run_async(
                            query_graph(
                                st.session_state.hybrid_rag,
                                prompt,
                                st.session_state.top_k,
                                st.session_state.query_type
                            )
                        )

                        # Display answer
                        st.write(answer)

                        # Display context in expander
                        with st.expander("📚 Retrieved Context"):
                            st.text(context)

                        # Add assistant response to messages
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "context": context
                        })

                    except Exception as e:
                        error_msg = f"❌ Query failed: {str(e)}"
                        st.error(error_msg)
                        logger.exception("Query execution error")

                        # Add error to messages
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": error_msg
                        })


if __name__ == "__main__":
    main()
