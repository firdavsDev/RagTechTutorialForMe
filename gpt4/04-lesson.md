## 🧠 4–Dars: RAG Arxitekturasi — Componentlar, Oqim va Dizayn Patterns


### 🎯 Maqsad:

* RAG arxitekturasi qanday qismlardan iboratligini tushunish
* Har bir komponentni funksiyasi va o‘zaro bog‘liqligini ko‘rish
* RAG tizimi ichida qanday oqim (pipeline) bo‘lishini o‘rganish

---

### 🏗 RAG Arxitekturasi – 3 Asosiy Qism

#### 1️⃣ **Encoder + Retriever**

Foydalanuvchi so‘rovini embedding qiladi va uni vector bazaga yuboradi

#### 2️⃣ **Document Retriever (Index + Store)**

Embeddings asosida mos hujjatlarni topadi (retrieval)

#### 3️⃣ **Generator / Reader**

Topilgan hujjatlar asosida javob generatsiya qiladi

---

### 🔧 Arxitektura Komponentlari (Advanced)

| Qism                    | Tavsif                                                    |
| ----------------------- | --------------------------------------------------------- |
| **Query Encoder**       | So‘rovni embedding’ga aylantiradi (BERT, MiniLM, MPNet)   |
| **Document Encoder**    | Har bir hujjatni embedding’ga aylantiradi                 |
| **Vector Store**        | Embedding’larni saqlovchi DB (FAISS, Pinecone, Weaviate)  |
| **Retriever**           | Vector bazadan eng yaqin hujjatlarni topadi               |
| **Reranker (optional)** | Natijalarni relevans bo‘yicha qayta tartiblaydi           |
| **Generator (Decoder)** | Hujjat asosida javob generatsiya qiladi (T5, GPT, LLaMA)  |
| **Prompt builder**      | Hujjat va so‘rov asosida prompt yaratadi                  |
| **Response Formatter**  | Generatsiya qilingan matnni foydalanuvchiga moslab beradi |

---

### ⚙️ Texnik Oqim (Step-by-Step)

```text
1. User query → embedding → retriever
2. Retriever → vector DB → top N documents
3. Top documents + query → prompt yaratish
4. Prompt → generator model (GPT, T5)
5. Output → final answer
```

```mermaid
graph TD
    A[User Query] --> B[Query Encoder]
    B --> C[Retriever]
    C --> D[Vector Store]
    D --> E[Relevant Docs]
    E --> F[Prompt Builder]
    F --> G[Generator (Decoder)]
    G --> H[Final Answer]
```

---

### 🧠 Prompting Strategy

RAG tizimlarida prompt yaratish bu san’at:

```text
You are a helpful assistant. Based on the following documents, answer the question.

Documents:
- Doc 1: ...
- Doc 2: ...
- Doc 3: ...

Question: {user_query}

Answer:
```

Kuchli prompting:

* kontekstdagi ma’lumotlarni ko‘rsatuvchi
* redundant chunklarni olib tashlovchi
* tartiblangan (relevant first)

---

### 🧪 Real hayotdagi RAG dizaynlar (3 Pattern)

#### 🧱 Pattern 1: **Classic RAG**

* So‘rov → retriever → top docs → generator
* Eng oddiy va keng tarqalgan

#### ⚡ Pattern 2: **Streaming RAG**

* Hujjatlar ko‘p bo‘lsa, stream qilib topiladi
* Caching, batching, async ishlatiladi

#### 🧠 Pattern 3: **Hybrid RAG**

* Sparse (BM25) + Dense retriever natijalari birlashtiriladi
* Yuqori relevans uchun reranker bilan qayta tartiblanadi

---

### 🧩 Bonus: Komponentlar orasida ishlash formatlari

| Component    | Input Format     | Output Format        |
| ------------ | ---------------- | -------------------- |
| Encoder      | Text → Embedding | 384 / 768 / 1024 vec |
| Vector Store | Embedding        | Top-N doc IDs + meta |
| Generator    | Prompt + context | Text response        |
| Retriever    | Query → Top docs | (Text, Metadata)     |

---

### ✅ Uyga vazifa:

1. `SentenceTransformers` yordamida oddiy query encoder yaratib ko‘r.
2. `FAISS` yordamida kichik vector store yasab ko‘r.
3. `prompt` ichiga 3 ta hujjatni qo‘shib, GPT orqali test qilib ko‘r.

