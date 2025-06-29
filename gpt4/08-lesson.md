## 🧠 8–Dars: Hallucination va Foydalanuvchi Ishonchini Oshirish – RAG’da Xavfsizlik va Aniqlik

---

### 🎯 Maqsad:

* **Hallucination** nima va qanday yuzaga keladi?
* RAG tizimida **javoblar ishonchliligini oshirish**
* Prompting, reranking va citation texnikalari bilan tanishish
* Real foydalanuvchilar uchun **"proof-based" output** berishni o‘rganish

---

## ❗️ Hallucination nima?

> **Hallucination** — AI model (GPT va boshqalar) real manba asosida bo‘lmagan, lekin to‘g‘riday eshitiladigan **yolg‘on javob** generatsiya qilishi.

RAG — bu muammoni **retrieved factual knowledge** orqali kamaytirish uchun yaratilgan. Lekin bu **100% kafolat** bermaydi.

---

## 🔍 RAG’da Hallucination qanday sodir bo‘ladi?

| Sabab                                  | Tushuntirish                                             |
| -------------------------------------- | -------------------------------------------------------- |
| ❌ Relevantsiz chunk tanlanadi          | Retriever noto‘g‘ri hujjat olib beradi                   |
| 🧠 Model contextni noto‘g‘ri tushunadi | Prompt noto‘g‘ri tuzilgan, token kesilishi               |
| 🚫 Context ko‘p, prompt limit oshadi   | Model kerakli bo‘lakni ko‘rmaydi                         |
| 🔁 Feedback yo‘q                       | Foydalanuvchi noto‘g‘ri javobni bildiradigan sistem yo‘q |

---

## 🔐 Ishonchli RAG qilish strategiyalari

### ✅ 1. **Citation-based Prompting**

Modeldan javobda **manba nomi yoki id** keltirishni so‘rash:

```text
Answer the question below using the context. Cite the source for every statement.

Context:
[1] "In 2024, internet speed in Uzbekistan reached 25 Mbps." (doc_id: 123)
[2] "Ookla reported a drop to 22 Mbps in rural areas." (doc_id: 124)

Question: What was Uzbekistan's average internet speed in 2024?

Answer:
According to source [1], Uzbekistan's average internet speed in 2024 was 25 Mbps.
```

### ✅ 2. **Chunk reranking (Relevance Scoring)**

Retriever topgan hujjatlarni **reranker model** orqali baholab:

* `BERTScore`, `CrossEncoder`, `MonoT5`, `Cohere reranker` dan foydalanish

```python
from sentence_transformers import CrossEncoder
model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
scores = model.predict([[query, doc] for doc in top_docs])
```

### ✅ 3. **Max token + answer grounding**

* Context token limitni belgilash (masalan, 3000 token)
* Promptda: "only answer using context" deb yozish
* Generated answer ichida `source: [doc_id]` formatida link chiqarish

### ✅ 4. **Guardrail qo‘llash (e.g., Rebuff, Guardrails.ai)**

* Javobda "manba topilmadi" yoki "noma’lum" degan javobni ruxsat berish
* Output validation qo‘shish: javobda `source` yo‘q bo‘lsa — userga ko‘rsatma berilmaydi

### ✅ 5. **Feedback loop**

* Foydalanuvchi "to‘g‘ri / noto‘g‘ri" deya baholasa — score’lar update qilinadi
* Click-through rate (CTR) asosida retrieverni moslashtirish

---

## 💡 Prompt misoli (anti-hallucination)

```text
Answer the question using ONLY the following documents.
If the answer is not present, say "I don't know based on the provided information."

Documents:
{chunk_1}
{chunk_2}
{chunk_3}

Question: {query}

Answer:
```

---

## 📈 Javob ishonchliligini oshirish bo‘yicha metrikalar

| Metrika              | Tavsifi                                   |
| -------------------- | ----------------------------------------- |
| **Factuality Score** | Javobdagi faktlar manbaga mosmi           |
| **Attribution Rate** | Javobdagi gaplarda source link bormi      |
| **Faithfulness**     | Contextdagi gapga qanchalik sodiq qolgan  |
| **Answer Coverage**  | Contextdagi asosiy faktlar qayd etilganmi |

---

## ✅ Uyga vazifa:

1. Citation bilan ishlovchi prompt tuz
2. Cross-encoder orqali `relevance score` chiqar
3. “I don’t know” fallback’li promptni test qilib ko‘r
