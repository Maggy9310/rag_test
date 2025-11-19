
#MongoDB Vector Search Chatbot with LangChain and Streamlit

# Standard library imports
import os
from operator import itemgetter

# Third-party imports
import streamlit as st
from pymongo import MongoClient

# LangChain imports
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
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
    database = client['reviews_second']
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
        text_key="review_text",
        index_name="vector_index",
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
    # If retriever returned documents, join their page_content
    if docs and len(docs) > 0:
        try:
            return "\n\n".join(getattr(doc, "page_content", str(doc)) for doc in docs)
        except Exception:
            return "\n\n".join(str(doc) for doc in docs)

    # Fallback: query the Mongo collection for any size_chart docs so context isn't empty
    try:
        fallback_cursor = collection.find({"doc_type": "size_chart"}).limit(5)
        fallback = [d.get("review_text") or str(d) for d in fallback_cursor]
        if fallback:
            return "\n\n".join(fallback)
    except Exception:
        pass

    return "No contextual documents found."


# ---------- Size-chart helpers using your Mongo data ----------
try:
    import numpy as np
    def _to_array(x):
        return np.asarray(x, dtype=float)
except Exception:
    np = None
    def _to_array(x):
        return [float(i) for i in x]


def upsert_size_docs(collection):
    """Idempotent upsert of a catalog of normalized size-chart documents (multiple brands)."""
    size_docs = [
        # Nike jackets
        {"brand":"Nike","category":"Jacket","doc_type":"size_chart","gender":"Male",
         "size_label":"S","chest_in":"34-36","height_in":"65-68","fit":"standard",
         "fit_notes":"regular","review_text":"Nike Jacket size S. Chest 34-36 in. Height 65-68 in."},
        {"brand":"Nike","category":"Jacket","doc_type":"size_chart","gender":"Male",
         "size_label":"M","chest_in":"38-40","height_in":"68-70","fit":"standard",
         "fit_notes":"regular","review_text":"Nike Jacket size M. Chest 38-40 in. Height 68-70 in."},
        {"brand":"Nike","category":"Jacket","doc_type":"size_chart","gender":"Male",
         "size_label":"L","chest_in":"42-44","height_in":"70-72","fit":"standard",
         "fit_notes":"regular","review_text":"Nike Jacket size L. Chest 42-44 in. Height 70-72 in."},

        # Adidas jackets
        {"brand":"Adidas","category":"Jacket","doc_type":"size_chart","gender":"Male",
         "size_label":"S","chest_in":"35-37","height_in":"65-68","fit":"athletic",
         "fit_notes":"slimmer cut","review_text":"Adidas Jacket size S. Chest 35-37 in. Height 65-68 in. Athletic fit."},
        {"brand":"Adidas","category":"Jacket","doc_type":"size_chart","gender":"Male",
         "size_label":"M","chest_in":"38-40","height_in":"68-70","fit":"athletic",
         "fit_notes":"slimmer cut","review_text":"Adidas Jacket size M. Chest 38-40 in. Height 68-70 in. Athletic fit."},

        # Uniqlo shirts
        {"brand":"Uniqlo","category":"Shirt","doc_type":"size_chart","gender":"Male",
         "size_label":"M","chest_in":"38-40","height_in":"68-70","fit":"standard",
         "fit_notes":"true to size","review_text":"Uniqlo Shirt size M. Chest 38-40 in. Height 68-70 in."},
        {"brand":"Uniqlo","category":"Shirt","doc_type":"size_chart","gender":"Female",
         "size_label":"S","chest_in":"32-34","height_in":"62-64","fit":"standard",
         "fit_notes":"true to size","review_text":"Uniqlo Shirt size S. Chest 32-34 in. Height 62-64 in."},

        # H&M dresses
        {"brand":"H&M","category":"Dress","doc_type":"size_chart","gender":"Female",
         "size_label":"M","chest_in":"34-36","height_in":"64-66","fit":"standard",
         "fit_notes":"varies by style","review_text":"H&M Dress size M. Chest 34-36 in. Height 64-66 in."},
    ]

    for d in size_docs:
        q = {"brand": d["brand"], "category": d["category"], "doc_type": d["doc_type"], "size_label": d["size_label"]}
        collection.update_one(q, {"$setOnInsert": d}, upsert=True)

    return f"Upserted {len(size_docs)} size docs."


def embed_missing_docs(collection, embeddings):
    """Embed docs missing an `embedding` field and write embedding back to MongoDB."""
    cursor = collection.find({"doc_type":"size_chart", "embedding": {"$exists": False}})
    docs = list(cursor)
    if not docs:
        return "No docs to embed."

    for doc in docs:
        text = doc.get("review_text") or ""
        if not text.strip():
            continue
        vec = embeddings.embed_query(text)
        # Store embedding as list (JSON serializable)
        collection.update_one({"_id": doc["_id"]}, {"$set": {"embedding": list(vec)}})

    return f"Embedded {len(docs)} docs."


def cosine_sim(a, b):
    """Compute cosine similarity; uses numpy when available, else pure-Python fallback."""
    if np is not None:
        a = _to_array(a)
        b = _to_array(b)
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)
    else:
        # pure-python fallback
        a_list = _to_array(a)
        b_list = _to_array(b)
        dot = sum(x * y for x, y in zip(a_list, b_list))
        norm_a = sum(x * x for x in a_list) ** 0.5
        norm_b = sum(x * x for x in b_list) ** 0.5
        denom = norm_a * norm_b
        if denom == 0:
            return 0.0
        return dot / denom


def local_candidate_search(collection, embeddings, query_text, filter_spec, top_k=5):
    """Fallback when Atlas index filtering isn't available: prefilter in MongoDB, then score by cosine locally."""
    candidates = list(collection.find(filter_spec))
    if not candidates:
        return []

    qvec = embeddings.embed_query(query_text)
    scored = []
    for doc in candidates:
        vec = doc.get("embedding")
        if not vec:
            continue
        sim = cosine_sim(qvec, vec)
        scored.append((sim, doc))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for s, d in scored[:top_k]]


def recommend_size_with_llm(llm, retrieved_docs, height_in, height_cm, weight_lb, weight_kg, bust_in, brand, category, gender):
    """Call the LLM with context and measurements to produce a size recommendation."""
    # Build context text robustly from different doc shapes (Document-like or dict)
    if retrieved_docs and len(retrieved_docs) > 0:
        parts = []
        for d in retrieved_docs:
            if hasattr(d, 'page_content'):
                parts.append(getattr(d, 'page_content'))
            elif isinstance(d, dict):
                parts.append(d.get('review_text') or str(d))
            else:
                parts.append(str(d))
        context = "\n\n".join(parts)
    else:
        context = ""
    user_question = f"Recommend a size for height {height_in} in ({height_cm} cm), weight {weight_lb} lb ({weight_kg} kg), bust {bust_in} in."

    # If context lacks numeric size facts, append a safe fallback size chart and guidance
    import re
    has_numeric = False
    if context:
        if re.search(r"\b(chest|bust)\b", context, flags=re.I) or re.search(r"\bheight\b", context, flags=re.I) or re.search(r"\d{2,3}\s*(cm|in|inches|lbs|kg)\b", context, flags=re.I):
            has_numeric = True

    # Analyze qualitative size signals in the context (e.g., "runs large", "swimming in it", "runs small")
    qualitative_patterns = {
        "runs_large": [r"runs? large", r"too big", r"big on", r"swimming", r"oversized", r"roomy", r"very large"],
        "runs_small": [r"runs? small", r"too small", r"tight", r"snug", r"small on"],
        "true_to_size": [r"true to size", r"fits? well", r"fits? perfectly", r"true-to-size"],
    }

    qual_counts = {k: 0 for k in qualitative_patterns}
    observation_sentences = []
    if context:
        lower_ctx = context.lower()
        for label, patterns in qualitative_patterns.items():
            for p in patterns:
                matches = re.findall(p, lower_ctx)
                if matches:
                    qual_counts[label] += len(matches)
                    # capture an example fragment for evidence
                    observation_sentences.append(f"Evidence: '{matches[0]}' found for {label}.")

    # Decide bias: -1 => recommend smaller, +1 => recommend larger, 0 => no bias
    bias = 0
    if qual_counts["runs_large"] > max(qual_counts["runs_small"], qual_counts["true_to_size"]):
        bias = -1
    elif qual_counts["runs_small"] > max(qual_counts["runs_large"], qual_counts["true_to_size"]):
        bias = 1

    observation_text = "\n".join(observation_sentences) if observation_sentences else "No qualitative size comments found."

    # Provide a short directive the LLM can follow based on qualitative evidence
    if bias == -1:
        bias_directive = "Multiple reviewers indicate the item runs large; when in doubt, recommend one size smaller than the chart suggests and explain why."
    elif bias == 1:
        bias_directive = "Multiple reviewers indicate the item runs small/tight; when in doubt, recommend one size larger than the chart suggests and explain why."
    else:
        bias_directive = "No consistent qualitative bias detected; follow numeric chart rules."

    # Append qualitative observation directive to context so the LLM accounts for it
    context = (context + "\n\nObservations:\n" + observation_text + "\nDirective:\n" + bias_directive) if context else ("Observations:\n" + observation_text + "\nDirective:\n" + bias_directive)

    # Define multiple fallback charts keyed by (Category, Gender). These are conservative default ranges (inches).
    # The LLM will be given the selected chart for the requested category/gender when no numeric facts exist in context.
    def _fmt_chart(name, ranges, height_hint=None, note=None):
        parts = [f"{name}:"]
        parts.append("; ".join(f"{k}: {v}" for k, v in ranges.items()))
        if height_hint:
            parts.append(f"Height guidance: {height_hint}")
        if note:
            parts.append(f"Note: {note}")
        return "\n".join(parts)

    fallback_charts = {}
    # Dresses (female) — chest/bust ranges; dresses often size by bust/waist
    fallback_charts[("Dress", "Female")] = _fmt_chart(
        "Dress (Female)",
        {"S": "32-34", "M": "34-36", "L": "36-38", "XL": "38-40"},
        height_hint="Short/tall adjustments: if user >170 cm consider length and silhouette; when between sizes choose larger for comfort",
        note="Use bust first, then height for length adjustments"
    )
    # Shirts / Tops (female)
    fallback_charts[("Shirt", "Female")] = _fmt_chart(
        "Shirt (Female)",
        {"XS": "30-32", "S": "32-34", "M": "34-36", "L": "36-38"},
        height_hint="If user is taller, consider size up for sleeve length",
        note="Prefer bust measurement when available"
    )
    # Shirts / Tops (male)
    fallback_charts[("Shirt", "Male")] = _fmt_chart(
        "Shirt (Male)",
        {"S": "36-38", "M": "38-40", "L": "42-44", "XL": "46-48"},
        height_hint="If between chest ranges and user is tall, consider larger size",
        note="Athletic vs relaxed fit may change choice"
    )
    # Jackets (male)
    fallback_charts[("Jacket", "Male")] = _fmt_chart(
        "Jacket (Male)",
        {"S": "35-37", "M": "38-40", "L": "41-43", "XL": "44-46"},
        height_hint="Jackets often sized for chest; if user is >6ft consider length adjustments",
        note="Recommend layering allowance if user wants room"
    )
    # Generic default (conservative)
    fallback_charts[("default", "default")] = _fmt_chart(
        "Default (All)",
        {"S": "34-36", "M": "38-40", "L": "42-44", "XL": "46-48"},
        height_hint="Use height+weight to estimate body type if bust missing",
        note="If unsure, prefer larger size for comfort"
    )

    # Normalize keys and pick the best matching fallback chart
    cat_key = (str(category).title() if category else "").strip()
    gen_key = (str(gender).title() if gender else "").strip()
    selected = None
    if (cat_key, gen_key) in fallback_charts:
        selected = fallback_charts[(cat_key, gen_key)]
    elif (cat_key, "") in fallback_charts:
        selected = fallback_charts[(cat_key, "")]
    else:
        selected = fallback_charts[("default", "default")]

    fallback_reference = (
        "SelectedFallbackChart:\n" + selected + "\n\n"
        "Rules:\n"
        "1) If user's bust/chest measurement is provided, choose the size whose range contains the measurement. If measurement is between ranges, recommend the larger size and note adjustment (e.g., size up for layering).\n"
        "2) If bust is not provided, use height+weight to estimate body type and then choose a conservative size (prefer a slightly larger size if unsure).\n"
        "3) When recommending, always state the numeric range used and any adjustment advice.\n"
    )

    if not has_numeric:
        # append selected fallback_reference to context so LLM can follow deterministic rules
        if context:
            context = context + "\n\n" + fallback_reference
        else:
            context = fallback_reference

    template = """You are an expert apparel fitter. Use ONLY the context facts and the user's measurements to recommend a size label and explain your reasoning.

Context: {context}
Brand: {brand}
Category: {category}
Gender: {gender}
User measurements: height {height_in} in ({height_cm} cm), weight {weight_lb} lb ({weight_kg} kg), bust {bust_in} in

Task: Provide a short recommendation that includes:
- Recommended size label (e.g., S, M, L)
- The numeric chest/bust range you used to decide
- Any adjustment advice (e.g., size up for layering or if measurements are borderline)

If the context includes the word "FallbackReference", follow the explicit rules in that section (default chest ranges and selection rules). Keep the answer concise and base it only on the provided context. If context is empty, explicitly say so and give a conservative recommendation.
"""

    prompt = ChatPromptTemplate.from_template(template)
    chain = (prompt | llm | StrOutputParser())
    out = chain.invoke({
        "context": context,
        "brand": brand,
        "category": category,
        "gender": gender,
        "height_in": str(height_in),
        "height_cm": str(height_cm),
        "weight_lb": str(weight_lb),
        "weight_kg": str(weight_kg),
        "bust_in": str(bust_in),
    })

    return out


@st.cache_resource
def get_vector_store():
    collection = get_mongodb_collection()
    embeddings, _ = get_embeddings_and_llm()
    return MongoDBAtlasVectorSearch(
        collection=collection,
        embedding=embeddings,
        text_key="review_text",
        index_name="vector_index",
        relevance_score_fn="cosine",
    )

# --------------------------------------------------------------

# Streamlit app config
st.set_page_config(
    page_title="AI Chat Assistant 2025",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
    <style>
    :root {
        /* Professional palette: muted navy + teal accent */
        --ink: #0f1724; /* dark slate */
        --bg: #f4f7fb; /* very light cool background */
        --ai: #eef9f8; /* subtle mint */
        --ai-ink: #0ea5a4; /* teal */
        --user: #ffffff; /* clean white for user */
        --user-ink: #0f1724;
        --accent: #e6f7f6; /* pale accent */
        --accent-2: #0ea5a4; /* accent teal */
        --muted: #64748b;
        --card-bg: #ffffff;
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
        background: linear-gradient(180deg, var(--accent), #f7fbfb);
        border-left: 3px solid rgba(15,23,36,0.08);
        box-shadow: none;
        padding: 18px;
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
        font-weight: 800;
        display: inline-block;
        background: linear-gradient(90deg,#ffffff,#f3fbfb);
        padding: 8px 14px;
        border: 1px solid rgba(15,23,36,0.06);
        border-radius: 10px;
        box-shadow: 0 6px 18px rgba(2,6,23,0.06);
        font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial;
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
    st.markdown("### 🤖 AI Chat Assistant")
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

# Size-recommendation helpers in the sidebar (controls to upsert/embed/retrieve)
with st.sidebar:
    st.markdown("---")
    st.markdown("### 🧭 Size Recommendation Tools")
    brand = st.selectbox("Brand", options=["Nike", "Adidas", "Other"])
    category = st.selectbox("Category", options=["Jacket", "Shirt", "Dress"])
    gender = st.selectbox("Gender", options=["Male", "Female", "Unisex"])

    if st.button("Upsert sample size docs"):
        msg = upsert_size_docs(collection)
        st.success(msg)

    if st.button("Embed missing size docs"):
        with st.spinner("Embedding missing docs — this uses your embeddings quota..."):
            msg = embed_missing_docs(collection, embeddings)
            st.success(msg)

    if st.button("Run size retrieval"):
        # Attempt filtered retriever first, fallback to local cosine on stored embeddings
        filter_spec = {"brand": brand, "category": category, "doc_type": "size_chart", "gender": gender}
        vector_store = get_vector_store()
        query_text = "Recommend a size for someone 6'2 180lbs"
        try:
            retr = vector_store.as_retriever(search_kwargs={"filter": filter_spec, "k": 5})
            docs = retr.get_relevant_documents(query_text)
            st.write("Retrieved via Atlas filtered retriever:")
            for d in docs:
                st.write(getattr(d, 'page_content', d))
        except Exception as e:
            st.warning("Filtered retriever failed — falling back to local search. Error: " + str(e))
            candidates = local_candidate_search(collection, embeddings, query_text, filter_spec, top_k=5)
            st.write("Retrieved via local cosine search:")
            for c in candidates:
                st.write(c.get("review_text") or c)

    # Measurements inputs for LLM recommendation
    st.markdown("---")
    st.markdown("### 📏 Measurements for Recommendation")
    height_in = st.number_input("Height (inches)", min_value=40.0, max_value=90.0, value=68.0, step=0.5)
    weight_lb = st.number_input("Weight (lbs)", min_value=40.0, max_value=400.0, value=160.0, step=1.0)
    bust_in = st.number_input("Chest/Bust (inches)", min_value=20.0, max_value=60.0, value=38.0, step=0.5)

    if st.button("Recommend size (LLM)"):
        filter_spec = {"brand": brand, "category": category, "doc_type": "size_chart", "gender": gender}
        vector_store = get_vector_store()
        # derive metric conversions for the LLM
        height_cm = round(height_in * 2.54, 1)
        weight_kg = round(weight_lb / 2.20462, 1)
        try:
            retr = vector_store.as_retriever(search_kwargs={"filter": filter_spec, "k": 5})
            docs = retr.get_relevant_documents(f"height {height_in} in ({height_cm} cm) weight {weight_lb} lb ({weight_kg} kg) bust {bust_in} in")
        except Exception as e:
            st.warning("Filtered retriever failed — falling back to local search. Error: " + str(e))
            docs = local_candidate_search(collection, embeddings, f"height {height_in} in ({height_cm} cm) weight {weight_lb} lb ({weight_kg} kg) bust {bust_in} in", filter_spec, top_k=5)

        with st.spinner("Asking the LLM for a size recommendation..."):
            recommendation = recommend_size_with_llm(llm, docs, height_in, height_cm, weight_lb, weight_kg, bust_in, brand, category, gender)
            st.markdown("**LLM Recommendation:**")
            st.write(recommendation)

# Main header
st.title("🤖 AI Chat Assistant")
st.markdown("### Ask me anything!")
st.markdown("---")

# Clothing-themed banner and quick category cards
st.markdown("""
<div style='display:flex;align-items:center;gap:18px;padding:18px;background:linear-gradient(90deg,#fff7f9,#f8fbfb);border-radius:14px;border:1px solid rgba(15,23,36,0.04)'>
    <div style='width:64px;height:64px;border-radius:12px;background:linear-gradient(135deg,var(--accent-2),#e6f7f6);display:flex;align-items:center;justify-content:center;color:white;font-weight:800;font-size:20px'>CF</div>
    <div>
        <div style='font-weight:700;font-size:20px;color:var(--ink)'>Clothing Fit Assistant</div>
        <div style='color:var(--muted);font-size:13px'>Professional size recommendations — based on brand size charts and user measurements.</div>
    </div>
    <div style='margin-left:auto;display:flex;gap:12px'>
        <div style='background:var(--card-bg); padding:10px 14px; border-radius:10px; box-shadow:0 2px 6px rgba(2,6,23,0.04);border:1px solid rgba(15,23,36,0.03)'>
            <div style='font-size:13px;font-weight:700;color:var(--ink)'>Jackets</div>
            <div style='font-size:12px;color:var(--muted)'>Layer-friendly</div>
        </div>
        <div style='background:var(--card-bg); padding:10px 14px; border-radius:10px; box-shadow:0 2px 6px rgba(2,6,23,0.04);border:1px solid rgba(15,23,36,0.03)'>
            <div style='font-size:13px;font-weight:700;color:var(--ink)'>Shirts</div>
            <div style='font-size:12px;color:var(--muted)'>Casual & formal</div>
        </div>
        <div style='background:var(--card-bg); padding:10px 14px; border-radius:10px; box-shadow:0 2px 6px rgba(2,6,23,0.04);border:1px solid rgba(15,23,36,0.03)'>
            <div style='font-size:13px;font-weight:700;color:var(--ink)'>Dresses</div>
            <div style='font-size:12px;color:var(--muted)'>Silhouettes</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


def get_response(user_query, chat_history):
    
    template = """Answer the user question based on the context provided.

    Chat history: {chat_history}
    Context: {context}
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
        AIMessage(content="Hello, welcome! How can I help you?"),
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




