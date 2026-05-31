import os
import json
import time
import asyncio
import sys
import pandas as pd
import streamlit as st
import chromadb

import nest_asyncio
nest_asyncio.apply()
from typing import Optional
# LlamaIndex
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core.schema import TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.groq import Groq as LlamaGroq

# Groq Official
from groq import Groq

# MCP
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from contextlib import AsyncExitStack

from dotenv import load_dotenv
load_dotenv()

# ====================== PAGE CONFIG ======================
st.set_page_config(
    page_title="Stock AI Assistant",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====================== CUSTOM CSS (FINANCIAL PREMIUM THEME) ======================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght=400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
}
.stApp {
    background: #0B0E14 !important; /* خلفية داكنة جداً مستوحاة من منصات التداول العالمية */
    color: #F4F6F8 !important;
}

[data-testid="stChatInput"] {
    border-radius: 12px !important;
    padding: 2px !important;
    background-color: transparent !important;
}

.stChatInput textarea {
    background-color: #1E2330 !important; /* خلفية رمادية داكنة مريحة ومتناسقة مع الثيم */
    color: #FFFFFF !important;            /* نص الكتابة باللون الأبيض الناصع لقراءة مثالية */
    font-size: 15px !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 10px !important;
    box-shadow: none !important;
}

.stChatInput textarea::placeholder {
    color: #9CA3AF !important; 
}

[data-testid="stChatInputSubmitButton"] {
    color: #10B981 !important; 
}

/* NAVBAR */
.navbar {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    z-index: 999;
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 15px 50px;
    background: rgba(11, 14, 20, 0.9);
    backdrop-filter: blur(16px);
    border-bottom: 1px solid rgba(255, 255, 255, 0.05);
}
.logo {
    font-size: 28px;
    font-weight: 800;
    letter-spacing: -0.5px;
    color: #FFFFFF;
}
.logo span { color: #10B981; }

/* HERO / HEADER SECTION */
.hero {
    min-height: 40vh;
    display: flex;
    justify-content: center;
    align-items: center;
    position: relative;
    overflow: hidden;
    padding: 110px 20px 30px 20px;
    background: linear-gradient(rgba(11, 14, 20, 0.88), rgba(11, 14, 20, 0.98)),
                url("https://images.unsplash.com/photo-1611974789855-9c2a0a7236a3?q=80&w=2070");
    background-size: cover;
    background-position: center;
    border-bottom: 1px solid rgba(255, 255, 255, 0.03);
}
.hero-content {
    position: relative;
    z-index: 5;
    text-align: center;
    max-width: 950px;
}
.badge {
    display: inline-block;
    padding: 6px 16px;
    border-radius: 100px;
    border: 1px solid rgba(16, 185, 129, 0.2);
    background: rgba(16, 185, 129, 0.06);
    color: #34D399;
    letter-spacing: 1px;
    margin-bottom: 18px;
    font-size: 11px;
    font-weight: 600;
}
.main-title {
    font-size: 48px;
    font-weight: 800;
    line-height: 1.2;
    margin-bottom: 15px;
    color: #FFFFFF;
    letter-spacing: -1px;
}
.highlight {
    background: linear-gradient(90deg, #10B981, #FBBF24);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.subtitle {
    color: #9CA3AF;
    font-size: 15px;
    line-height: 1.6;
    max-width: 750px;
    margin: auto;
}
</style>
""", unsafe_allow_html=True)

# ====================== NAVBAR & HERO ======================
st.markdown("""
<div class="navbar">
    <div class="logo">Stock <span>AI</span></div>
    <div class="menu">📊</div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <div class="hero-content">
        <div class="badge">CONVERSATIONAL INTELLIGENCE PLATFORM</div>
        <h1 class="main-title">
            Next-Gen STOCK AI <br>
            <span class="highlight">AI Chatbot Assistant</span>
        </h1>
        <p class="subtitle">
            Ask questions, analyze market trends, and query financial documents in real-time. 
            Powered by Groq LLMs, LlamaIndex, Advanced RAG, and MCP Tools execution.
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

# ====================== CONFIG ======================
Groq_api_key = os.getenv("GROQ_API_KEY", "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")
embedding_model = 'sentence-transformers/all-MiniLM-L6-v2'
groq_model = 'llama-3.3-70b-versatile'

DEFAULT_SERVER_PATH = r"C:\Users\ELZAHBIA\Vs_code\my_mcp_server.py"

# Initialize uploaded files in session state if not existing
if "uploaded_files_dict" not in st.session_state:
    st.session_state.uploaded_files_dict = {}

# ====================== SIDEBAR (MULTI-FILE UPLOADER) ======================
with st.sidebar:
    st.header("📊 Data Management")
    # Added accept_multiple_files=True here
    uploaded_files = st.file_uploader("Upload your CSV datasets", type=["csv"], accept_multiple_files=True)
    
    if uploaded_files:
        current_files_dict = {}
        for file in uploaded_files:
            try:
                # Cache dataframes locally to prevent re-reading on every cycle
                df_temp = pd.read_csv(file)
                current_files_dict[file.name] = df_temp
                st.success(f" Loaded: **{file.name}** ({len(df_temp)} rows)")
            except Exception as e:
                st.error(f"Failed to read {file.name}: {e}")
        
        st.session_state.uploaded_files_dict = current_files_dict
    else:
        st.session_state.uploaded_files_dict = {}
        st.info("💡 No files uploaded. Active mode: General Chat & MCP Server Tools.")

# ====================== RESOURCES ======================
@st.cache_resource
def init_llama_resources():
    Settings.llm = LlamaGroq(
        model=groq_model,
        api_key=Groq_api_key,
        temperature=0
    )
    Settings.embed_model = HuggingFaceEmbedding(model_name=embedding_model)

    cache_path = os.path.join(os.getcwd(), "cache_db_analyst")
    os.makedirs(cache_path, exist_ok=True)
    
    chroma_client = chromadb.PersistentClient(path=cache_path)
    collection = chroma_client.get_or_create_collection("semantic_cache")
    
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex(
        nodes=[], 
        storage_context=storage_context,
        embed_model=Settings.embed_model
    )
    return index

try:
    cache_index = init_llama_resources()
except Exception as e:
    st.error(f"Resource Initialization Error: {e}")
    st.stop()

# ====================== SEMANTIC CACHE ======================
def update_cache(query: str, answer: str):
    node = TextNode(
        text=query,
        metadata={
            'answer': str(answer),
            'timestamp': time.time(),
            'is_valid': True
        }
    )
    cache_index.insert_nodes([node])

def check_semantic_cache(query: str, threshold: float = 0.85):
    MAX_TTL = 24 * 60 * 60
    retriever = cache_index.as_retriever(similarity_top_k=1)
    try:
        results = retriever.retrieve(query)
        if results and results[0].score >= threshold:
            node = results[0].node
            metadata = node.metadata
            age = time.time() - metadata.get("timestamp", 0)
            if metadata.get("is_valid", True) and age <= MAX_TTL:
                return metadata["answer"], "fresh"
    except Exception:
        pass
    return None, "miss"

# ====================== MCP CLIENT ======================
class MCPClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.grok = Groq(api_key=Groq_api_key)

    async def connect_to_server(self, server_script_path: str):
        if not os.path.exists(server_script_path):
            raise FileNotFoundError(f"server not exist {server_script_path}")

        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")

        command = sys.executable if is_python else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=os.environ.copy()
        )

        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
        await self.session.initialize()

    async def process_query(self, query: str) -> str:
        # فحص الكاش الدلالي
        cached, status = check_semantic_cache(query)
        if cached:
            return f"**[Cache Hit]** {cached}"

        # بناء السياق لجميع الملفات المرفوعة ديناميكياً
        system_context = ""
        if st.session_state.uploaded_files_dict:
            system_context = "The user has uploaded multiple datasets:\n"
            for file_name, df_local in st.session_state.uploaded_files_dict.items():
                cols = list(df_local.columns)
                sample_data = df_local.head(3).to_dict(orient='records')
                system_context += f"- File Name: {file_name}\n"
                system_context += f"  Columns: {cols}\n"
                system_context += f"  3-row Sample Data: {json.dumps(sample_data)}\n\n"
            system_context += "Answer queries about these datasets carefully matching filenames and contents provided above.\n\n"

        messages = [
            {"role": "system", "content": f"{system_context}You are an expert data analyst assistant. Answer clearly and look at the provided context or files metadata carefully."},
            {"role": "user", "content": query}
        ]

        if not self.session:
            # Fallback فوري ومباشر في حال غياب اتصال السيرفر لضمان جلب النتائج من الملف المرفوع
            response = self.grok.chat.completions.create(
                model=groq_model,
                messages=messages,
                temperature=0.2
            )
            actual_answer = response.choices[0].message.content
            update_cache(query, actual_answer)
            return actual_answer

        try:
            mcp_tools_resp = await self.session.list_tools()
            groq_formatted_tools = []
            for tool in mcp_tools_resp.tools:
                groq_formatted_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.inputSchema
                    }
                })

            response = self.grok.chat.completions.create(
                model=groq_model,
                messages=messages,
                tools=groq_formatted_tools if groq_formatted_tools else None,
                temperature=0.2
            )

            assistant_message = response.choices[0].message
            final_outputs = []

            if assistant_message.content:
                final_outputs.append(assistant_message.content)

            if assistant_message.tool_calls:
                messages.append(assistant_message)
                
                for tool_call in assistant_message.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)

                    result = await self.session.call_tool(tool_name, tool_args)
                    result_str = "".join([block.text for block in result.content if hasattr(block, 'text')])

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_name,
                        "content": result_str
                    })

                final_response = self.grok.chat.completions.create(
                    model=groq_model,
                    messages=messages
                )
                final_outputs.append(final_response.choices[0].message.content)

            actual_answer = "\n\n".join(final_outputs)
        except Exception:
            response = self.grok.chat.completions.create(
                model=groq_model,
                messages=messages,
                temperature=0.2
            )
            actual_answer = response.choices[0].message.content

        update_cache(query, actual_answer)
        return actual_answer

    async def cleanup(self):
        if self.session:
            await self.exit_stack.aclose()


# ====================== CHAT INTERFACE (STREAMLIT) ======================
st.markdown("<br>", unsafe_allow_html=True)
st.subheader("💬 Chat AI Assistant (General & Data Analysis)")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask me anything, or upload datasets on the sidebar to analyze them!"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            
            async def run_mcp_pipeline():
                client = MCPClient()
                try:
                    if not os.path.exists(DEFAULT_SERVER_PATH):
                        return await client.process_query(prompt)
                    
                    try:
                        await client.connect_to_server(DEFAULT_SERVER_PATH)
                    except Exception:
                        pass
                        
                    res = await client.process_query(prompt)
                    return res
                except Exception as e:
                    return f"❌ Error during execution: {e}"
                finally:
                    await client.cleanup()

            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            response = loop.run_until_complete(run_mcp_pipeline())
            
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

st.caption("Note: Dashboard is fully operational. Upload data on the left panel anytime to transition into analysis mode.")
#"I have uploaded my movies dataset. Please analyze it and find the top 3 highest-rated or most popular Horror movies in the file. After that, let's assume a production company wants to invest their box office profits into the stock market: check the stock NVDA and give me a full investment recommendation for it based on your tools
