## 🧠 6–Dars: Retriever va Generator’ni Ulaymiz — Prompting, Chunking, Context Injection

---

### 🎯 Maqsad:

* Retriever topgan ma’lumotni modelga qanday qilib **samarali tarzda uzatish**ni o‘rganish
* **Prompt dizayni**, **context injection**, va **chunking strategiyalari** bilan tanishish
* Real chatbotlar va QA systemalarda nima ishlaydi, nima ishlamaydi — bilib olish

---

## 🔗 1. Retriever → Generator ulanishi

RAG arxitekturasi bo‘yicha:

```mermaid
graph TD
    A[User query] --> B[Retriever]
    B --> C[Top-k docs]
    C --> D[Prompt constructor]
    D --> E[Generator (e.g., GPT)]
    E --> F[Answer]
```

### ⚠️ Asosiy muammo:

* LLM’lar (GPT, T5) **konkret input uzunligi bilan cheklangan** (`context window`, masalan: GPT-4 = \~32K token)
* Bizda esa ko‘p hujjatlar bo‘lishi mumkin

Shuning uchun biz **chunking + prompt design** strategiyalarini ishlatamiz.

---

## ✂️ 2. Chunking strategiyalari

### 🔸 Nima bu “chunking”?

Katta hujjatlarni kichik bo‘laklarga bo‘lib, retrieval va generation bosqichida **token cheklovlariga moslab** ishlatish.

### 🌟 Eng ko‘p ishlatiladigan strategiyalar:

| Nomi                | Tavsifi                                                  |
| ------------------- | -------------------------------------------------------- |
| **Fixed-size**      | Har chunk 512/1024 token                                 |
| **Sliding window**  | Har chunk 70% avvalgisini o‘z ichiga oladi (overlap)     |
| **Semantic split**  | Matnni sentence/topic asosida bo‘lish                    |
| **Recursive split** | Paragraph → sentence → token tarzida sekin-sekin bo‘lish |

### 📌 Python (LangChain bilan chunking):

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_text(large_document)
```

---

## ✍️ 3. Prompt Dizayni (Prompt Engineering)

Prompt — bu generator modelga beriladigan **matnli kontekst + savol**.

### 📘 Prompt strukturasi:

```text
Answer the following question based on the documents provided.

Context:
- {doc_1}
- {doc_2}
- ...
- {doc_k}

Question: {user_query}
Answer:
```

### 📈 Yaxshi promptlar:

* Relevant hujjatlarni tartibli beradi (avval eng dolzari)
* Har bir hujjatni cheklangan token miqdorida tozalab beradi
* Tokenlarni behuda isrof qilmaydi (noisy matnni olib tashlaydi)

### 👨‍💻 Prompt Template misoli (Python):

```python
prompt_template = f"""
You are a helpful assistant.

Use the following context to answer the question:
{retrieved_docs}

Question: {query}
Answer:"""
```

---

## 🧠 4. Context Injection — Smart Strategy

Ko‘p holatda, 3–5 hujjat yetarli bo‘ladi. Lekin agar hujjatlar ko‘p bo‘lsa:

* **Top-N** tanlash: faqat eng relevant 3–5 hujjatni ol
* **Dynamic reranking**: retriever natijalarini score asosida tartibla
* **Max token limit**: har hujjatdan faqat 400–500 token kesib ol

---

## 🔄 Advanced: Iterative RAG

1. Prompt 1 → javob topilmadi
2. Rewriting query (e.g., "rephrase") + new retrieval
3. Final generation

Bunday oqim “multi-hop reasoning” uchun ishlatiladi (LangChain’dagi `Refine`, `MapReduce` flow).

---

## ✅ Uyga vazifa:

1. `RecursiveCharacterTextSplitter` bilan PDF matnni chunklab ko‘r.
2. Har bir chunkni embedding qilib FAISS index yarat.
3. So‘rovga relevant chunklarni olib, GPT prompt tuzib, test qil.

---
