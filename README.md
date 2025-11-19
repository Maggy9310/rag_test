# SmartSize Recommendation - MongoDB Vector Search Chatbot

A modern, Gen-Z styled chatbot built with Streamlit that uses MongoDB Atlas Vector Search and Google Gemini AI to provide intelligent, context-aware responses. Perfect for customer service, order management, and conversational AI applications.

![Python](https://img.shields.io/badge/python-3.12+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)
![LangChain](https://img.shields.io/badge/langchain-latest-green.svg)

## ✨ Features

- 🔍 **Vector Search**: Leverages MongoDB Atlas Vector Search for semantic document retrieval
- 🤖 **AI-Powered**: Uses Google Gemini 2.5 Flash for intelligent responses
- 💬 **Gen-Z Friendly**: Responses include modern slang and emojis
- 🎨 **Neobrutalist UI**: Bold, colorful design with a modern aesthetic
- 📊 **Session Tracking**: Monitor conversation metrics in real-time
- 🔗 **LangChain RAG**: Retrieval-Augmented Generation for accurate, context-aware answers
- 📈 **LangSmith Tracing**: Track and debug AI interactions
- 💾 **Persistent Chat**: Conversation history maintained throughout session
 - 🅰️ **Font**: Uses the modern 'Inter' sans-serif font (loaded via Google Fonts) for a clean, readable UI

## 🏗️ Architecture

This application uses:
- **Frontend**: Streamlit with custom CSS styling
- **Vector Database**: MongoDB Atlas with vector search capabilities
- **Embeddings**: Google Generative AI Embeddings (gemini-embedding-001)
- **LLM**: Google Gemini 2.5 Flash
- **Framework**: LangChain for RAG pipeline
- **Monitoring**: LangSmith for tracing and debugging

## 📋 Prerequisites

- Python 3.12 or higher
- MongoDB Atlas account with Vector Search enabled
- Google AI API key
- LangSmith account (optional, for tracing)

## 🚀 Installation

1. **Clone the repository**
```bash
git clone https://github.com/quan3010/rag_test
cd rag_test
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up Streamlit secrets**

Create a `.streamlit/secrets.toml` file with the following structure and **replace your API keys** accordingly:

```toml
[langsmith]
tracing = "true"
endpoint = "https://api.smith.langchain.com"
api_key = "your-langsmith-api-key"

[google]
api_key = "your-google-api-key"

[mongodb]
uri = "mongodb+srv://user1:user1@cluster0.lqirl.mongodb.net/?retryWrites=true&w=majority"
```



## 🎯 Usage

1. **Run the Streamlit app**
```bash
streamlit run app.py
```

2. **Open your browser** to `http://localhost:8501`

3. **Start chatting!** Ask questions about your documents and the AI will respond with relevant information.

## 🎨 Features in Detail

### Chat Interface
- Clean, modern chat interface with distinct styling for AI and user messages
- Real-time streaming responses
- Message history preserved during session

### Sidebar Controls
- **Session Info**: Track number of messages sent
- **Clear Chat**: Reset conversation history
- **About Section**: Quick reference to technologies used

### RAG Pipeline
The application uses a sophisticated RAG (Retrieval-Augmented Generation) pipeline:
1. User query is embedded using Google's embedding model
2. Vector search retrieves relevant documents from MongoDB
3. Context and chat history are passed to Gemini AI
4. AI generates contextual, accurate responses

## 🛠️ Customization

### Modify the AI Personality
Edit the `template` in the `get_response()` function to change the chatbot's behavior:
```python
template = """You are a helpful customer assistant...
Your custom instructions here...
"""
```

### Change Styling
Modify the custom CSS in the Streamlit app to adjust colors and design:
```python
st.markdown("""
    <style>
    :root {
        --ink: #111111;
        --bg: #f7f7f7;
        /* Add your custom colors */
    }
    </style>
""", unsafe_allow_html=True)
```

### Vector Search Configuration
Adjust retriever parameters in `get_retriever()`:
```python
vector_store = MongoDBAtlasVectorSearch(
    collection=collection,
    embedding=embeddings,
    index_name="vector_index_1",
    relevance_score_fn="cosine"  # or "euclidean", "dotProduct"
)
```

## 📊 Monitoring

The app integrates with LangSmith for comprehensive tracing:
- View all LLM calls and responses
- Debug retrieval quality
- Monitor latency and performance
- Analyze conversation flows

Access your traces at [smith.langchain.com](https://smith.langchain.com)

## 🔒 Security Notes

- Never commit `.streamlit/secrets.toml` to version control
- Add `.streamlit/` to your `.gitignore`
- Use environment variables in production
- Rotate API keys regularly

## 🐛 Troubleshooting

**Connection Issues**
- Verify MongoDB connection string and IP whitelist
- Check API keys are valid and have proper permissions

**Vector Search Not Working**
- Ensure vector index is created with correct dimensions (3072 for gemini-embedding-001)
- Verify documents have embeddings in the correct field

**Slow Responses**
- Check network latency to MongoDB Atlas
- Consider using a closer MongoDB region
- Optimize retriever parameters

## 📝 License

[Your License Here]

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

[Your Contact Information]

---

Made with ❤️ using Streamlit, LangChain, MongoDB Atlas, and Google Gemini AI
