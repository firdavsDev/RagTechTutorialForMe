
## 🧠 5–Dars: Dense vs Sparse Retrieval — FAISS, BM25, ScaNN va boshqalar

---

### 🎯 Maqsad:

* Retrievalning ikki asosiy turi — Sparse va Dense haqida chuqur tushunish
* Ularning farqlari, kuchli va zaif tomonlarini o‘rganish
* Qaysi vector search vositalari qanday ishlashini bilish

---

### 🔍 Retrieval nima?

Retriever — foydalanuvchi so‘rovi asosida ma’lumotlar bazasidan **eng mos hujjatlarni topib beradigan** mexanizm.

U ikki turga bo‘linadi:

| Turi       | Tushuntiruvchi nomlar                  |
| ---------- | -------------------------------------- |
| **Sparse** | Term-based search (e.g., BM25, TF-IDF) |
| **Dense**  | Semantic search (e.g., BERT + FAISS)   |

---

## 📌 1. Sparse Retrieval

**Asos:** So‘zlar (term) o‘z-o‘zini ifodalaydi. Hech qanday semantik ma’no yo‘q.
🧠 Model yod olmaydi — bu klassik “keyword match”.

### 🌟 Mashhur algoritmlar:

* `BM25` (Best Matching 25) — TF-IDFning optimized versiyasi

### ➕ Yaxshi tomonlari:

* Juda tez
* Oson sozlanadi
* To‘g‘ridan-to‘g‘ri matnga mos keladi (agar exact keyword bo‘lsa)

### ➖ Yomon tomonlari:

* Sinonim, semantik, kontekstni tushunmaydi
* So‘zlar shakli o‘zgarsa, tushunmaydi
* O‘zbek tilida yaxshi ishlashi qiyin (lemmatization yetishmaydi)

### 👨‍💻 Python’da BM25:

```python
from rank_bm25 import BM25Okapi

tokenized_corpus = [doc.split() for doc in documents]
bm25 = BM25Okapi(tokenized_corpus)
scores = bm25.get_scores(query.split())
```

---

## 📌 2. Dense Retrieval

**Asos:** Matn semantikasini embedding vektorga aylantirib, o‘xshashlikka qarab qidiradi.

### 🌟 Mashhur retriever arxitekturalari:

* `DPR` (Dense Passage Retrieval, Facebook)
* `Contriever`, `ColBERT`, `GTR`, `MiniLM`, `MPNet`

### ➕ Yaxshi tomonlari:

* Sinonimlar, semantik o‘xshashlikni tushunadi
* Har xil tilda juda yaxshi ishlaydi
* Zero-shot / multilingual ko‘rinishda ishlaydi

### ➖ Yomon tomonlari:

* Trening kerak (yoki pre-trained model kerak)
* Vector bazani qurish va xizmat qilish biroz murakkab
* Faiss/Pinecone kabi backendlar kerak

### 👨‍💻 Python’da Dense Retrieval (SentenceTransformers + FAISS):

```python
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')
corpus_embeddings = model.encode(documents)
index = faiss.IndexFlatL2(corpus_embeddings.shape[1])
index.add(np.array(corpus_embeddings))

query_vector = model.encode(["your query"])
D, I = index.search(np.array([query_vector]), k=5)
```

---

## 🥊 Dense vs Sparse — Taqqoslash

| Xususiyat                   | Sparse (BM25) | Dense (DPR, Faiss)               |
| --------------------------- | ------------- | -------------------------------- |
| Training kerakmi?           | Yo‘q          | Ha / Pretrained                  |
| Sinonim tushunadimi?        | Yo‘q          | Ha                               |
| Speed                       | Tez           | Sekinroq, ammo indexing mumkin   |
| Lokal so‘zlar bilan ishlash | Zo‘r          | Yaxshi, lekin tokenization muhim |
| Index narxi                 | Juda arzon    | Biroz qimmat (vektorli)          |

---

## 🔧 Dense Retrieval uchun mashhur vositalar

| Vosita       | Tavsifi                                         |
| ------------ | ----------------------------------------------- |
| **FAISS**    | Facebook tomonidan yaratilgan, GPU-ni qo‘llaydi |
| **ScaNN**    | Google dan — scalable va tez, hybrid dense      |
| **Annoy**    | Spotify dan, RAM-da ishlaydi, minimal setup     |
| **Qdrant**   | Rust asosidagi vector DB, REST API bor          |
| **Pinecone** | SaaS — tayyor platforma                         |
| **Weaviate** | Semantic + graph, local & cloud                 |

---

## 🚀 Advanced Kombinatsiyalar

### 🔀 Hybrid Retrieval

Sparse + Dense natijalarini birga ishlatish:

* `BM25 + Dense`
* `Weighted ensemble`: scorelarni birlashtirib top N olasan

### 🔁 Feedback loop bilan fine-tune

* Foydalanuvchi javob bergan feedback asosida retriever’ni yaxshilash

---

### ✅ Uyga vazifa:

1. HuggingFace’dan `sentence-transformers/all-mpnet-base-v2` ni ishlatib FAISS index qur.
2. `rank_bm25` bilan natijani solishtir.
3. `retrieval hybrid strategy` haqida maqola o‘qib chiq (HuggingFace blogda bor).
