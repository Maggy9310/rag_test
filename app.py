"""
MongoDB Vector Search Chatbot with LangChain and Streamlit
"""

# Standard library imports
import os
from operator import itemgetter

# Third-party imports
import streamlit as st
from pymongo import MongoClient

# LangChain imports
from langchain.chat_models import init_chat_model
from langchain-classic.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch

# Load environment variables
# LangSmith / LangChain
os.environ["LANGCHAIN_TRACING_V2"] = st.secrets["langsmith"]["tracing"]
os.environ["LANGCHAIN_ENDPOINT"] = st.secrets["langsmith"]["endpoint"]
os.environ["LANGCHAIN_API_KEY"] = st.secrets["langsmith"]["api_key"]

# Google
os.environ["GOOGLE_API_KEY"] = st.secrets["google"]["api_key"]

# Mongo
os.environ["MONGO_URI"] = st.secrets["mongodb"]["uri"]

@st.cache_resource
def get_mongodb_collection():
    """Connect to MongoDB and return collection (cached)."""
    MONGODB_URI = os.environ['MONGO_URI']
    client = MongoClient(MONGODB_URI)
    database = client['demo_vector_db']
    return database['reviews']

@st.cache_resource
def get_embeddings_and_llm():
    """Initialize embeddings and LLM models (cached)."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai")
    return embeddings, llm

@st.cache_resource
def get_retriever():
    """Initialize vector store retriever (cached)."""
    collection = get_mongodb_collection()
    embeddings, _ = get_embeddings_and_llm()
    
    vector_store = MongoDBAtlasVectorSearch(
        collection=collection,
        embedding=embeddings,
        index_name="vector_index_1",
        relevance_score_fn="cosine"
    )
    return vector_store.as_retriever()

# Initialize cached resources
collection = get_mongodb_collection()
embeddings, llm = get_embeddings_and_llm()
retriever = get_retriever()

# Helper function
def format_docs(docs):
    """Format retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)

# Streamlit app config
st.set_page_config(
    page_title="AI Chat Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
    <style>
    :root {
        --ink: #111111;
        --bg: #f7f7f7;
        --ai: #D9F2FF;
        --ai-ink: #00C2FF;
        --user: #FFF0D6;
        --user-ink: #FF8A00;
        --accent: #FFE500;
        --accent-2: #00C2FF;
    }

    /* App background */
    .stApp { background: var(--bg); }

    /* Chat messages */
    .stChatMessage {
        background: #ffffff !important;
        border: 3px solid var(--ink);
        border-radius: 14px;
        padding: 1rem;
        margin: 0.75rem 0;
        box-shadow: 8px 8px 0 var(--ink);
    }

    /* AI and Human variations */
    .stChatMessage[data-testid*="ai"] {
        background: var(--ai) !important;
        border-color: var(--ai-ink);
        box-shadow: 8px 8px 0 var(--ai-ink);
    }
    
    .stChatMessage[data-testid*="user"] {
        background: var(--user) !important;
        border-color: var(--user-ink);
        box-shadow: 8px 8px 0 var(--user-ink);
    }

    /* Input */
    .stChatInputContainer {
        background: #fff;
        border-top: 0;
        padding-top: 1rem;
    }
    
    .stChatInputContainer textarea {
        background: #ffffff !important;
        color: var(--ink) !important;
        border: 3px solid var(--ink) !important;
        border-radius: 12px;
        box-shadow: 6px 6px 0 var(--ink);
    }
    
    .stChatInputContainer textarea:focus {
        outline: none !important;
        border-color: var(--accent-2) !important;
        box-shadow: 6px 6px 0 var(--accent-2);
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: var(--accent);
        border-left: 3px solid var(--ink);
        box-shadow: -8px 0 0 var(--ink);
    }
    
    [data-testid="stSidebar"] * {
        color: var(--ink) !important;
    }

    /* Buttons */
    .stButton button {
        background: var(--accent-2) !important;
        color: var(--ink) !important;
        border: 3px solid var(--ink) !important;
        border-radius: 12px;
        box-shadow: 6px 6px 0 var(--ink);
        transition: transform 0.1s ease, box-shadow 0.1s ease;
    }
    
    .stButton button:hover {
        transform: translate(-2px, -2px);
        box-shadow: 8px 8px 0 var(--ink);
    }

    /* Titles */
    h1 {
        color: var(--ink);
        text-align: left;
        font-weight: 900;
        display: inline-block;
        background: var(--accent);
        padding: 6px 12px;
        border: 3px solid var(--ink);
        border-radius: 12px;
        box-shadow: 6px 6px 0 var(--ink);
    }
    
    h3 {
        color: #333;
        text-align: left;
        font-weight: 700;
    }

    /* Metrics */
    [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {
        color: var(--ink) !important;
    }

    /* Divider */
    hr {
        border: 0;
        height: 3px;
        background: var(--ink);
        box-shadow: 4px 4px 0 #ffd400;
        opacity: 1;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 🤖 AI Chat Assistant in gen-Z slang")
    st.markdown("---")
    
    st.markdown("#### 📊 Session Info")
    if "chat_history" in st.session_state:
        msg_count = len([m for m in st.session_state.chat_history if isinstance(m, HumanMessage)])
        st.metric("Messages Sent", msg_count)
    
    st.markdown("---")
    st.markdown("#### ⚙️ Settings")
    
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = [
            AIMessage(content="Hello! I'm your AI assistant. How can I help you today?")
        ]
        st.rerun()
    
    st.markdown("---")
    st.markdown("#### 💡 About")
    st.markdown("""
    This chatbot uses:
    - 🔍 MongoDB Atlas Vector Search
    - 🤖 Google Gemini AI
    - 🔗 LangChain RAG
    - 📊 LangSmith Tracing
    """)
    
    st.markdown("---")
    st.markdown("##### Made with ❤️ using Streamlit")

# Main header
st.title("🤖 AI Chat Assistant")
st.markdown("### Ask me anything about your documents!")
st.markdown("---")


def get_response(user_query, chat_history):
    
    template = """You are a helpful customer assistant to help with orders. 
    Answer the following questions considering the history and
    the context of your restaurant. User gen-z slang here and there.
    Use emojis where appropriate.
    If you don't know the answer, just say you don't know.
    Do not make up answers.
    Ask follow-up questions to have customers order more items.
    Once done, confirm the order summary and total price.
    Ask how they would like to pay and provide payment options.

    Chat history: {chat_history}
    Context of your restaurant: {context}
    User question: {question}
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    chain = (
        {
            "context": itemgetter("question") | retriever | format_docs,
            "question": itemgetter("question"),
            "chat_history": itemgetter("chat_history"),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain.stream({
        "chat_history": chat_history,
        "question": user_query,
    })

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello, I am a bot. How can I help you?"),
    ]

# Display conversation history
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)

# Handle user input
user_query = st.chat_input("Type your message here...")
if user_query is not None and user_query != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        response = st.write_stream(get_response(user_query, st.session_state.chat_history))

    st.session_state.chat_history.append(AIMessage(content=response))




