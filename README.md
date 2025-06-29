
# RagChatBot

### Loyiha maqsadi
Veb-sayt va 10,000+ PDF/DOC hujjatlar asosida intelligent chatbot yaratish.

### Texnologiyalar stack'i
- **Backend**: Python, FastAPI
- **Vector Database**: Chroma/Pinecone
- **LLM**: OpenAI GPT-4 / Ollama (local)
- **Embeddings**: OpenAI text-embedding-ada-002
- **Document Processing**: LangChain
- **Frontend**: Streamlit/Gradio

### Sistem arxitekturasi

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   User Query    │    │   Vector DB     │    │   LLM API       │
│   Interface     │────│   (Chroma)      │────│   (OpenAI)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │              ┌─────────────────┐              │
         └──────────────│  RAG Pipeline   │──────────────┘
                        │   Controller    │
                        └─────────────────┘
                                 │
                    ┌─────────────────────┐
                    │  Document Store     │
                    │  (10,000+ files)    │
                    └─────────────────────┘
```
