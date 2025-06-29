## 🚀 1–Dars: RAG nima? Nega kerak?

### 🧠 Qisqacha:

RAG (Retrieval-Augmented Generation) — bu generativ model (masalan, ChatGPT) va ma’lumot izlash mexanizmini (retriever) birlashtiruvchi texnika. Bu yondashuv modelga real dunyo ma'lumotlaridan foydalanishga yordam beradi, ayniqsa **model bilmaydigan narsalarni** tushuntirish uchun foydali.

### 🧩 Asosiy komponentlar:

1. **Retriever (izlovchi)** — foydalanuvchi bergan so‘rovga mos hujjatlarni qidirib topadi.
2. **Reader/Generator** — topilgan hujjatlar asosida foydalanuvchiga javob yozadi.

### 🎯 Misol:

> So‘rov: “Uzbekistonda 2024-yilda eng ko‘p ishlatilgan banklar qaysilar?”

Agar bu modelga o‘rgatilmagan bo‘lsa, u to‘g‘ri javob bera olmaydi. Lekin retriever — Google qidiruvi yoki PDFdan ma’lumot olib, generativ modelga beradi. Shunda u aniq faktlar asosida javob beradi.

---

### ✅ RAG nega muhim?

| Aspekt                 | RAG                                       |
| ---------------------- | ----------------------------------------- |
| *Knowledge Up-to-date* | Ha, chunki tashqi ma'lumotdan foydalanadi |
| *Train qilish narxi*   | Arzonroq, fine-tuningdan                  |
| *Scalability*          | Yaxshi, chunki content tez yangilanadi    |
| *Hallucination*        | Kamroq, agar yaxshi retriever bo‘lsa      |

---

### 🎓 Uyga vazifa:

1. “RAG vs Fine-tuning” farqini o‘zingcha yozib ko‘r.
2. Quyidagi so‘zlarni izlab ko‘r: `Retrieval-based QA`, `open-domain QA`, `Dense retriever`, `RAG paper (Lewis et al, 2020)`.

---
### 📚 Qo‘shimcha o‘qishlar:
- [RAG paper (Lewis et al, 2020)](https://arxiv.org/abs/2005.11401)
- [RAG vs Fine-tuning](https://www.example.com/rag-vs-fine-tuning)