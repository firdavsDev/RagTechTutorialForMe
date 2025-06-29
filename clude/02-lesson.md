# RAG Arxitekturasi va Komponentlari

## Dars maqsadi
Ushbu darsda siz RAG tizimining batafsil arxitekturasini, har bir komponentning vazifasini va ular o'rtasidagi bog'lanishni o'rganasiz.

## RAG tizimining umumiy arxitekturasi

```
[Foydalanuvchi so'rovi] 
         ↓
[Query Processing] 
         ↓
[Retrieval System] ← [Vector Database]
         ↓
[Context Ranking]
         ↓
[Prompt Construction]
         ↓
[Language Model (LLM)]
         ↓
[Response Generation]
         ↓
[Javob filtrlash va formatlash]
         ↓
[Final Response]
```

## 1. Data Ingestion Pipeline (Ma'lumot Kiritish Quvuri)

### **1.1 Document Loading**
- Turli formatdagi hujjatlarni yuklash (PDF, DOCX, TXT, HTML)
- Strukturali va strukturasiz ma'lumotlarni qabul qilish
- Real-time va batch processing rejimlarini qo'llab-quvvatlash

### **1.2 Document Preprocessing**
- **Text Extraction:** Hujjatlardan matnni ajratib olish
- **Cleaning:** Keraksiz belglar, formatting, noise ni tozalash
- **Normalization:** Matnni standart formatga keltirish
- **Language Detection:** Til aniqlash va tegishli processors ni tanlash

### **1.3 Text Chunking**
- **Fixed-size chunking:** Belgilangan o'lchamdagi bo'laklar
- **Semantic chunking:** Ma'no bo'yicha bo'lish
- **Sliding window:** Ustma-ust tushadigan bo'laklar
- **Hierarchical chunking:** Daraja-daraja bo'lish

### **1.4 Embedding Generation**
- Har bir chunk uchun vektor tasvirini yaratish
- Embedding modellarini tanlash (BERT, Sentence-BERT, OpenAI)
- Batch processing va optimization

### **1.5 Vector Storage**
- Embedding vectorlarni saqlash
- Metadata bilan bog'lash
- Index yaratish va optimizatsiya

## 2. Query Processing Component

### **2.1 Query Understanding**
```python
# Pseudocode
def process_query(user_query):
    # Intent aniqlash
    intent = detect_intent(user_query)
    
    # Entity extraction
    entities = extract_entities(user_query)
    
    # Query expansion
    expanded_query = expand_query(user_query)
    
    # Query embedding
    query_vector = generate_embedding(expanded_query)
    
    return query_vector, intent, entities
```

### **2.2 Query Expansion Texnikalari**
- **Synonyms expansion:** Sinonimlar qo'shish
- **Related terms:** Bog'liq terminlar qidirish
- **Historical queries:** Oldingi so'rovlar tahlili
- **Domain-specific expansion:** Sohaga xos kengaytirish

## 3. Retrieval System (Qidiruv Tizimi)

### **3.1 Similarity Search**
```python
# Vector similarity qidiruv
def retrieve_documents(query_vector, top_k=5):
    # Cosine similarity hisoblash
    similarities = calculate_cosine_similarity(
        query_vector, 
        document_vectors
    )
    
    # Top-k natijalarni tanlash
    top_documents = select_top_k(similarities, k=top_k)
    
    return top_documents
```

### **3.2 Retrieval Strategiyalari**
- **Dense Retrieval:** Neural embedding asosida
- **Sparse Retrieval:** TF-IDF, BM25 asosida
- **Hybrid Retrieval:** Dense va sparse birlashma
- **Multi-vector Retrieval:** Turli xil vectorlar

### **3.3 Filtering va Re-ranking**
- **Relevance filtering:** Tegishlilik bo'yicha filtrlash
- **Diversity promotion:** Xilma-xillikni ta'minlash
- **Temporal filtering:** Vaqt bo'yicha filtrlash
- **Authority scoring:** Manbaning ishonchliligi

## 4. Context Construction

### **4.1 Context Selection**
```python
def construct_context(retrieved_docs, max_tokens=2000):
    context_parts = []
    token_count = 0
    
    for doc in retrieved_docs:
        doc_tokens = count_tokens(doc.content)
        
        if token_count + doc_tokens <= max_tokens:
            context_parts.append(doc.content)
            token_count += doc_tokens
        else:
            # Truncate yoki skip
            break
    
    return "\n\n".join(context_parts)
```

### **4.2 Context Optimization**
- **Token limit management:** Token cheklovini boshqarish
- **Information density:** Ma'lumot zichligini oshirish
- **Redundancy removal:** Takrorlanishni olib tashlash
- **Coherence maintenance:** Izchillikni saqlash

## 5. Prompt Engineering Component

### **5.1 Prompt Template**
```
System: Siz yordamchi assistentsiz. Berilgan kontekst asosida aniq va foydali javob bering.

Context:
{retrieved_context}

Question: {user_question}

Instructions:
- Faqat berilgan kontekst asosida javob bering
- Agar javob kontekstda yo'q bo'lsa, "Ma'lumot mavjud emas" deb ayting
- Manbani ko'rsating
- Qisqa va aniq javob bering

Answer:
```

### **5.2 Prompt Optimization**
- **Role definition:** Rol aniq belgilash
- **Context formatting:** Kontekstni formatlash
- **Instruction clarity:** Ko'rsatmalarni aniqlashtirish
- **Output structure:** Chiqish formatini belgilash

## 6. Generation Component

### **6.1 LLM Integration**
```python
def generate_response(prompt, model="gpt-3.5-turbo"):
    response = llm_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1,  # Deterministik javob uchun
        max_tokens=500
    )
    
    return response.choices[0].message.content
```

### **6.2 Generation Parameters**
- **Temperature:** Kreativlik darajasi (0.0-1.0)
- **Max tokens:** Maksimal javob uzunligi
- **Top-p:** Nucleus sampling parametri
- **Frequency penalty:** Takrorlanishni kamaytirish

## 7. Response Post-processing

### **7.1 Quality Control**
```python
def validate_response(response, context):
    # Hallucination detection
    is_grounded = check_grounding(response, context)
    
    # Relevance check
    is_relevant = check_relevance(response, user_query)
    
    # Safety filter
    is_safe = safety_filter(response)
    
    if not (is_grounded and is_relevant and is_safe):
        return generate_fallback_response()
    
    return response
```

### **7.2 Response Enhancement**
- **Citation adding:** Manba ko'rsatish qo'shish
- **Formatting:** Chiroyli formatlash
- **Confidence scoring:** Ishonch darajasini belgilash
- **Follow-up suggestions:** Qo'shimcha savollar taklif qilish

## 8. Feedback Loop va Learning

### **8.1 User Feedback Collection**
- **Explicit feedback:** To'g'ridan-to'g'ri baholash
- **Implicit feedback:** Click-through, dwell time
- **Correction feedback:** Foydalanuvchi tuzatishlari

### **8.2 System Improvement**
- **Retrieval optimization:** Qidiruv sifatini yaxshilash
- **Ranking adjustment:** Ranking algoritmlarini sozlash
- **Model fine-tuning:** Modelni yanada o'qitish

## RAG Pipeline misoli

```python
class RAGPipeline:
    def __init__(self):
        self.document_store = VectorDatabase()
        self.retriever = DenseRetriever()
        self.generator = LLMGenerator()
    
    def process_query(self, query):
        # 1. Query processing
        processed_query = self.preprocess_query(query)
        
        # 2. Document retrieval
        retrieved_docs = self.retriever.retrieve(
            processed_query, 
            top_k=5
        )
        
        # 3. Context construction
        context = self.construct_context(retrieved_docs)
        
        # 4. Prompt creation
        prompt = self.create_prompt(query, context)
        
        # 5. Response generation
        response = self.generator.generate(prompt)
        
        # 6. Post-processing
        final_response = self.post_process(response)
        
        return final_response
```

## Performance Metrics

### **Retrieval Metrics:**
- **Recall@K:** Top-K da to'g'ri hujjatlar nisbati
- **Precision@K:** Qaytarilgan hujjatlarning aniqligi
- **MRR:** Mean Reciprocal Rank
- **NDCG:** Normalized Discounted Cumulative Gain

### **Generation Metrics:**
- **ROUGE:** Overlap-based evaluation
- **BLEU:** N-gram precision
- **BERTScore:** Semantic similarity
- **Human evaluation:** Inson baholashi

## Dars xulosasi

RAG tizimi murakkab pipeline bo'lib, har bir komponent muhim rol o'ynaydi. Tizimning samaradorligi barcha komponentlarning uyg'un ishlashiga bog'liq.

## Keyingi darsga tayyorgarlik

Keyingi darsda biz vektor bazalar va embedding texnologiyalari haqida batafsil gaplashamiz.

## Amaliy vazifa

1. RAG pipeline ning qaysi qismi eng muhim deb o'ylaysiz va nima uchun?
2. Qanday holatlarda retrieval sifati yomonlashishi mumkin?
3. Context window cheklovi muammosini qanday hal qilish mumkin?