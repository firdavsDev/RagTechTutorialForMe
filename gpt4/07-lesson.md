## 🧠 7–Dars: Vector DB tanlash — FAISS, Qdrant, Weaviate, Pinecone, Annoy, ScaNN

---

### 🎯 Maqsad:

* Vector DB (embedding bazalari) qanday ishlashini tushunish
* Qaysi holatda qaysi DB eng to‘g‘ri variant ekanligini bilish
* FAISS, Qdrant, Weaviate, Pinecone, Annoy, ScaNN’ni **kod**, **speed**, **dev UX** bo‘yicha taqqoslash

---

## 📦 Vector DB nima?

Vector DB — bu matnlarni (embedding vektorlarni) **qidirish**, **tartiblash** va **metadata bilan saqlash** imkonini beradigan maxsus bazadir.

### ✅ Core features:

* Approximate Nearest Neighbor (ANN) search
* Vector + Metadata storage
* Top-K qidiruv (vektorlar orasidagi eng yaqinlar)
* Filtering (taglar, lang, user\_id, ...)

---

## 🔍 Tanlov mezonlari

| Mezoni                | Nima degani?                                                        |
| --------------------- | ------------------------------------------------------------------- |
| 🚀 Speed              | Nechta embedding orasidan top-N ni qancha tez topadi                |
| 🧠 Accuracy           | Haqiqatda “semantik yaqin” bo‘lganini topish darajasi               |
| 🛠 Dev UX             | API’si qulaymi? REST, SDK bormi? Docker bormi?                      |
| 🔁 Update support     | Real-time hujjat qo‘shish, o‘chirish imkoniyati                     |
| 🧩 Metadata filtering | Faqat `lang=uz`, `doc_type="news"` bo‘yicha filter qilish mumkinmi? |
| 💰 Hosting            | Localda ishlaydimi? Yoki faqat SaaSmi?                              |
| 🔐 Security           | Auth/ACL bor yo‘qmi (API keys, tokens)?                             |

---

## ⚙️ Keng tarqalgan Vector DBlar taqqoslash (2025)

| DB           | Hosting     | Speed 🔥 | Dev UX 😎 | Filter 🧩 | Update 🔁 | Qisqacha tavsif                        |
| ------------ | ----------- | -------- | --------- | --------- | --------- | -------------------------------------- |
| **FAISS**    | Local       | 🔥🔥🔥   | 😐        | ❌         | ❌         | Open-source, lekin oddiy, filter yo‘q  |
| **Qdrant**   | Local+Cloud | 🔥🔥     | 🔥🔥🔥    | ✅         | ✅         | Rustda yozilgan, REST + SDK, efficient |
| **Weaviate** | Local+Cloud | 🔥🔥     | 🔥🔥🔥    | ✅         | ✅         | Graph + semantic search + modular UX   |
| **Pinecone** | SaaS only   | 🔥🔥🔥   | 🔥🔥      | ✅         | ✅         | Tayyor servis, to‘lovli, scale zo‘r    |
| **Annoy**    | Local       | 🔥       | 😐        | ❌         | ❌         | RAM-based, oddiy, lekin light          |
| **ScaNN**    | Local       | 🔥🔥🔥   | 😐        | ❌         | ❌         | Google tomonidan, faqat qidiruv uchun  |

---

## 🧪 Real ishlatish holatlari

### ☑️ FAISS – eng mashhur local variant

```python
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')
docs = ["hello world", "this is about Uzbekistan", "GPT is a model"]
vecs = model.encode(docs)
index = faiss.IndexFlatL2(vecs.shape[1])
index.add(np.array(vecs))
D, I = index.search(model.encode(["Uzbekistan"]), k=2)
```

### ☑️ Qdrant – devlar uchun ideal

```bash
docker run -p 6333:6333 qdrant/qdrant
```

```python
from qdrant_client import QdrantClient
client = QdrantClient(host='localhost', port=6333)
client.upsert(collection_name='docs', points=[...])
```

### ☑️ Weaviate – semantic + graph qidiruvlar uchun

* Modular: text2vec, transformer, reranker pluginlar bilan
* GraphQL orqali query yozish mumkin

### ☑️ Pinecone – katta scale uchun

```python
import pinecone
pinecone.init(api_key="your-key", environment="us-west1-gcp")
index = pinecone.Index("my-index")
index.upsert([("id1", vec1), ("id2", vec2)])
```

---

## 💡 Qaysi holatda qaysi biri?

| Loyihangiz turi                            | Tavsiya etilgan DB   |
| ------------------------------------------ | -------------------- |
| 📘 Kurs sayt, PDF chatbot, blog search     | Qdrant / FAISS       |
| 🏢 Enterprise API, kerakli ACL, monitoring | Pinecone             |
| 🤖 Smart FAQ / Knowledge graph             | Weaviate             |
| 🧪 Test / Dev bosqichi                     | FAISS / Annoy        |
| 📦 Offline RAG bot                         | FAISS (RAMda yaxshi) |

---

### ✅ Uyga vazifa:

1. FAISS bilan embedding index tuz (yoki Qdrant docker’da ishga tushir)
2. 100+ hujjatni vector qilib saqla va `Top-5`ni qidirib ko‘r
3. `Qdrant`, `Pinecone`, `Weaviate` developer docslarini solishtir

