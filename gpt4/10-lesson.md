## 🧠 10–Dars: Custom Prompt, Context Filtering va Chat History bilan RAG-ni Yanada Aqlli Qilish

---

### 🎯 Maqsad:

* Har bir foydalanuvchi uchun **ancha individual, kontekstga bog‘liq javoblar** yaratish
* Chat history asosida **ko‘p bosqichli savol-javoblarni** boshqarish
* Prompt’ni moslashtirib **yanada kuchli, ishonchli RAG javoblar** olish
* Filtering orqali **specific documentlar** bilan ishlash

---

## 🧩 1. Custom Prompt tuzish (LangChain templates)

LLM’ga ko‘rsatma qanday bo‘lsa — javob shunga yarasha bo‘ladi. Prompt’ni to‘g‘ri yozish orqali RAG’da:

* Javob aniqligini oshirasan
* Manba kerakmi yo‘qmi — boshqarasan
* “hallucination” ni kamaytirasan

### 📘 Misol Prompt Template (Citation + Strict Answer):

```python
from langchain.prompts import PromptTemplate

prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
Siz ilmiy javob beruvchi yordamchisiz.

Faqat pastdagi matndan foydalangan holda savolga javob bering. Agar aniq javob bo‘lmasa, "Ma'lumot yo'q" deb yozing.

Context:
{context}

Savol: {question}

Javob:
"""
)
```

LangChain’da `RetrievalQA`ni bu prompt bilan ulang:

```python
from langchain.chains import RetrievalQA

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt_template}
)
```

---

## 🧠 2. Chat History (Memory) qo‘shish

Multi-turn savollar uchun, masalan:

```
User: O‘zbekistonning Aholisi qancha edi 2023da?
Bot: 36 million atrofida

User: Bu avvalgidan ko‘payganmi?
Bot: ...
```

Yaxshi RAG tizimi oldingi gapni ham inobatga oladi.

### ➕ LangChain `ConversationBufferMemory`

```python
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

chat_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory
)
```

### ❓ Chat orqali so‘rov:

```python
res = chat_chain({"question": "2023da aholi soni qancha edi?"})
res2 = chat_chain({"question": "Bu avvalgidan ko‘payganmi?"})
```

---

## 🧠 3. Filtering bilan contextni toraytirish (Metadata)

Agar hujjatlarda **tag**, `lang`, `document_type`, `source`, `owner_id` bo‘lsa — faqat kerakli bo‘lakni RAGga ber.

```python
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 5,
        "filter": {
            "lang": "uz",
            "type": "faq"
        }
    }
)
```

✅ Shunday qilib:

* RAG faqat O‘zbek tilidagi, FAQ turidagi matndan qidiradi.
* Boshqa til, eski fayllar ishtirok etmaydi.

---

## 🧠 4. RAG+Chatbot kombinatsiyasi – umumiy shakl

```python
User → Chat UI → LangChain (Retriever + Prompt + History) → LLM → Javob
```

Buni:

* TelegramBot bilan bog‘lashing mumkin
* FastAPI orqali backend qilishing mumkin
* Gradio yoki Streamlit bilan demo holatga olib chiqishing mumkin

---

## ✅ Uyga vazifa

1. Custom prompt bilan ishlaydigan RAG pipeline yasab ko‘r
2. Chat history’ni saqlovchi memory qo‘sh
3. Filtering bo‘yicha faqat `lang='uz'` hujjatlarni ishlatadigan RAG sozla
4. (bonus) Streamlit yoki FastAPI bilan RAG frontend qil
