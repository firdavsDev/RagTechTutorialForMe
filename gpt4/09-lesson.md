## 🧠 9–Dars: LangChain bilan End-to-End RAG Tizimi Qurish

---

### 🎯 Maqsad:

* `LangChain` framework’ida **to‘liq RAG tizimi** yaratish
* PDF yoki matnli faylni yuklab, uni retriever + generator bilan bog‘lash
* Faiss bilan local vector bazadan qidiruv qilish
* GPT-4 yoki OpenAI modeli bilan javob generatsiya qilish

---

## 📦 RAG tizimi oqimi (LangChain versiyasi)

```mermaid
graph TD
    A[Fayl (PDF / txt)] --> B[Chunking & Embedding]
    B --> C[FAISS index]
    D[User Query] --> E[Retriever (Top-N)]
    C --> E
    E --> F[Prompt yaratish]
    F --> G[OpenAI / LLM]
    G --> H[Javob chiqarish]
```

---

## ⚙️ 1. Muhitni sozlash

```bash
pip install langchain openai faiss-cpu tiktoken sentence-transformers
```

👉 OpenAI API key kerak:

```bash
export OPENAI_API_KEY=sk-...
```

---

## 📂 2. Asosiy modullarni import qilish

```python
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
```

---

## 📄 3. Hujjatni yuklash va chunklash

```python
loader = TextLoader("my_docs.txt", encoding='utf-8')
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_documents(docs)
```

---

## 🧠 4. Embedding + FAISS index yaratish

```python
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = FAISS.from_documents(chunks, embeddings)
db.save_local("rag_index")
```

---

## 🔍 5. Retriever va LLM’ni bog‘lash

```python
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})
llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0)

rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # "stuff" = all docs in one prompt
    retriever=retriever,
    return_source_documents=True
)
```

---

## ❓ 6. So‘rov yuborish

```python
query = "O'zbekistonning internet tezligi haqida nima deyilgan?"
result = rag_chain(query)

print("Javob:")
print(result['result'])

print("\nManbalar:")
for doc in result['source_documents']:
    print("-", doc.metadata.get("source", "Noma’lum"))
```

---

## ✅ Endi Senda Ishlaydigan RAG Tizim Bor

### Foydalanuvchi:

> "GPT haqida nima deyilgan?"

### RAG tizimi:

> "GPT bu OpenAI tomonidan ishlab chiqilgan generativ model bo‘lib, kontekst asosida javob bera oladi."

### Source:

* my\_docs.txt → `chunk_2`

---

## 🔥 Keyingi level (o‘rganish uchun):

* Vector DB sifatida Qdrant ishlatish (LangChain qo‘llab-quvvatlaydi)
* Prompt template’ni sozlash
* Feedback loop yoki reranking qo‘shish
* PDF, DOCX, YouTube transcript kabi murakkab loader’lar

---

## ✅ Uyga vazifa:

1. `.txt` yoki `.pdf` fayldan RAG pipeline qur
2. RAG bilan savol so‘rab, manbani chiqaruvchi bot yarat
3. `chain_type="map_reduce"` yoki `refine` modellarini test qilib ko‘r
