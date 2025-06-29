# Vektor Bazalar va Embedding Texnologiyalari

## Dars maqsadi
Ushbu darsda siz embedding vectorlar, ularning yaratilishi, saqlash usullari va vektor bazalarning turlari haqida chuqur bilim olasiz.

## 1. Embedding nima va nima uchun kerak?

### **1.1 Embedding tushunchasi**
Embedding - bu matn, so'z yoki hujjatning matematik vektor ko'rinishida ifodalanishi. Bu vectorlar semantik ma'noni sonlar orqali aks ettiradi.

```python
# Misol: So'zlar embeddingda
"qirol" → [0.2, -0.1, 0.8, 0.3, -0.5, ...]  # 768 o'lchamli vektor
"shoh"  → [0.18, -0.09, 0.82, 0.28, -0.48, ...] # O'xshash ma'no, o'xshash vektor
"olma"  → [-0.3, 0.7, -0.2, 0.9, 0.1, ...]   # Butunlay boshqa ma'no
```

### **1.2 Vektor fazoda semantik bog'liqlik**
```
Vector Space da masofalar:
- Yaqin vectorlar = O'xshash ma'no
- Uzoq vectorlar = Farqli ma'no
- Cosine similarity = Ma'noviy o'xshashlik o'lchovi

distance("qirol", "shoh") = 0.05    # Juda yaqin
distance("qirol", "olma") = 0.89    # Juda uzoq
```

## 2. Embedding Modellarining Turlari

### **2.1 Word-level Embeddings**

#### **Word2Vec (2013)**
```python
# Arxitektura
CBOW: Context → Target word
Skip-gram: Target word → Context

# Misol
"Men [MASK] yaxshi ko'raman kitob"
Context: [men, yaxshi, ko'raman, kitob] → Target: [o'qishni]
```

#### **GloVe (Global Vectors)**
- Global statistika asosida
- Co-occurrence matrix ishlatadi
- Word2Vec dan tezroq training

#### **FastText**
- Subword information ishlatadi
- OOV (Out-of-vocabulary) muammosini hal qiladi
- Morfologik boyitishlar

### **2.2 Sentence-level Embeddings**

#### **Sentence-BERT (SBERT)**
```python
# SBERT arxitekturasi
Input: "Bu juda yaxshi kitob"
       ↓
[CLS] Bu juda yaxshi kitob [SEP]
       ↓
BERT Encoder (12 layers)
       ↓
Pooling Layer (Mean/CLS/Max)
       ↓
768-dimensional vector
```

**Afzalliklari:**
- Sentence-level semantic understanding
- Efficient similarity calculation
- Pre-trained models mavjud

#### **Universal Sentence Encoder (USE)**
```python
# Google tomonidan ishlab chiqilgan
# Transformer va DAN arxitekturalari
# 512-dimensional vectors
# 16+ tillarda ishlaydi
```

### **2.3 Document-level Embeddings**

#### **Doc2Vec**
```python
# Paragraph Vector yondashuvi
# Document ID ni ham o'qitadi
# Variable-length documents uchun

def doc2vec_training():
    # Har bir document uchun unique ID
    doc_id = "DOC_001"
    context = ["bu", "muhim", "hujjat"]
    target = "juda"
    
    # Document vector + word vectors → target prediction
```

#### **BERT-based Document Embeddings**
```python
# Long documents uchun strategiyalar:

# 1. Chunking + Averaging
def document_embedding_chunked(document, chunk_size=512):
    chunks = split_into_chunks(document, chunk_size)
    chunk_embeddings = [bert_encode(chunk) for chunk in chunks]
    document_embedding = np.mean(chunk_embeddings, axis=0)
    return document_embedding

# 2. Sliding Window
def document_embedding_sliding(document, window_size=512, stride=256):
    windows = create_sliding_windows(document, window_size, stride)
    window_embeddings = [bert_encode(window) for window in windows]
    return aggregate_embeddings(window_embeddings)
```

## 3. Advanced Embedding Texnikalari

### **3.1 Multilingual Embeddings**

#### **mBERT (Multilingual BERT)**
```python
# 104 tilda pre-training
# Cross-lingual transfer learning
# O'zbek tili uchun ham ishlatish mumkin

# Misol
english_query = "What is artificial intelligence?"
uzbek_doc = "Sun'iy intellekt - bu mashinalarning..."

# mBERT orqali o'xshashlik hisoblash mumkin
```

#### **Sentence Transformers Multilingual**
```python
from sentence_transformers import SentenceTransformer

# Multilingual model yuklash
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Turli tillarda embeddinglar
sentences = [
    "This is a good book",           # English
    "Bu yaxshi kitob",               # Uzbek  
    "Это хорошая книга",            # Russian
]

embeddings = model.encode(sentences)
# Bir xil vector space da joylashadi
```

### **3.2 Domain-specific Embeddings**

#### **Scientific Domain**
```python
# BioBERT - tibbiyot uchun
# SciBERT - ilmiy maqolalar uchun
# FinBERT - moliyaviy matnlar uchun

# Custom domain adaptation
def adapt_to_domain(base_model, domain_corpus):
    # Continue pre-training on domain data
    adapted_model = continue_pretraining(
        model=base_model,
        corpus=domain_corpus,
        epochs=3
    )
    return adapted_model
```

### **3.3 Dense vs Sparse Embeddings**

#### **Dense Embeddings**
```python
# BERT, Sentence-BERT
# Pros: Semantic understanding, compact
# Cons: Requires neural models, black box

vector_dense = [0.23, -0.45, 0.67, 0.12, ...]  # 768 dimensions
```

#### **Sparse Embeddings**
```python
# TF-IDF, BM25
# Pros: Interpretable, efficient, exact matching
# Cons: No semantic understanding

vector_sparse = {
    "kitob": 0.67,
    "yaxshi": 0.45,
    "o'qish": 0.23,
    # qolgan 50,000 so'z uchun 0 qiymat
}
```

## 4. Vektor Bazalar (Vector Databases)

### **4.1 Vektor Bazalarning Turlari**

#### **In-Memory Solutions**
```python
# Faiss (Facebook AI Similarity Search)
import faiss
import numpy as np

# Index yaratish
dimension = 768
index = faiss.IndexFlatIP(dimension)  # Inner Product

# Vectorlarni qo'shish
embeddings = np.random.random((1000, dimension)).astype('float32')
index.add(embeddings)

# Qidiruv
query_vector = np.random.random((1, dimension)).astype('float32')
distances, indices = index.search(query_vector, k=5)
```

#### **Persistent Vector Databases**

**Pinecone**
```python
import pinecone

# Pinecone initialization
pinecone.init(api_key="your-api-key", environment="us-west1-gcp")

# Index yaratish
index_name = "rag-documents"
pinecone.create_index(
    name=index_name,
    dimension=768,
    metric="cosine"
)

# Ma'lumot kiritish
index = pinecone.Index(index_name)
index.upsert([
    ("doc_1", embedding_1, {"title": "Kitob 1", "author": "Muallif"}),
    ("doc_2", embedding_2, {"title": "Kitob 2", "author": "Muallif"})
])

# Qidiruv
results = index.query(
    vector=query_embedding,
    top_k=5,
    include_metadata=True
)
```

**Weaviate**
```python
import weaviate

# Client yaratish
client = weaviate.Client("http://localhost:8080")

# Schema belgilash
schema = {
    "class": "Document",
    "vectorizer": "text2vec-transformers",
    "properties": [
        {"name": "title", "dataType": ["string"]},
        {"name": "content", "dataType": ["text"]},
        {"name": "author", "dataType": ["string"]}
    ]
}
client.schema.create_class(schema)

# Ma'lumot qo'shish
client.data_object.create({
    "title": "Sun'iy Intellekt",
    "content": "AI haqida batafsil ma'lumot...",
    "author": "Olim Karimov"
}, "Document")

# Semantic qidiruv
result = client.query.get("Document", ["title", "content"]).with_near_text({
    "concepts": ["mashinali o'rganish"]
}).with_limit(5).do()
```

**Chroma**
```python
import chromadb

# Client yaratish
client = chromadb.Client()

# Collection yaratish
collection = client.create_collection(
    name="documents",
    embedding_function=embedding_function
)

# Ma'lumotlar qo'shish
collection.add(
    documents=["Bu birinchi hujjat", "Bu ikkinchi hujjat"],
    metadatas=[{"type": "article"}, {"type": "blog"}],
    ids=["id1", "id2"]
)

# Qidiruv
results = collection.query(
    query_texts=["qiziqarli maqola"],
    n_results=2
)
```

### **4.2 Indexing Strategiyalari**

#### **Flat Index (Brute Force)**
```python
# Har bir query uchun barcha vectorlar bilan taqqoslash
# O(n) complexity
# Kichik datasets uchun ideal
# Eng yuqori aniqlik
```

#### **IVF (Inverted File)**
```python
# Clustering asosida
# Datasets ni klasterlarga bo'lish
# Faqat yaqin klasterlardan qidirish
# O(√n) complexity

import faiss
index = faiss.IndexIVFFlat(quantizer, dimension, nlist=100)
index.train(training_data)  # Clustering uchun training
```

#### **HNSW (Hierarchical Navigable Small World)**
```python
# Graph-based approach
# Hierarchical structure
# O(log n) complexity
# Yuqori sifat va tezlik

import faiss
index = faiss.IndexHNSWFlat(dimension, M=16)  # M = connections per node
```

#### **Product Quantization (PQ)**
```python
# Memory-efficient compression
# Vector dimensionni kichik bo'laklarga bo'lish
# Har bir bo'lakni quantize qilish
# 8x-32x memory tejash

index = faiss.IndexPQ(dimension, M=8, nbits=8)  # 8 subvectors, 8 bits each
```

## 5. Embedding Quality Evaluation

### **5.1 Intrinsic Evaluation**

#### **Word Similarity Tasks**
```python
# Word pairs bilan test
def evaluate_word_similarity(embedding_model):
    word_pairs = [
        ("qirol", "shoh", 0.9),      # High similarity
        ("qirol", "olma", 0.1),      # Low similarity
        ("katta", "kichik", 0.2)     # Antonyms
    ]
    
    correlations = []
    for word1, word2, human_score in word_pairs:
        model_score = cosine_similarity(
            embedding_model[word1], 
            embedding_model[word2]
        )
        correlations.append((model_score, human_score))
    
    return pearson_correlation(correlations)
```

#### **Analogy Tasks**
```python
# "qirol" - "erkak" + "ayol" = "malika"
def analogy_test(embedding_model):
    analogies = [
        ("qirol", "erkak", "ayol", "malika"),
        ("katta", "kattalik", "kichik", "kichiklik")
    ]
    
    correct = 0
    for a, b, c, expected_d in analogies:
        # Vector arithmetic
        result_vector = (embedding_model[a] - 
                        embedding_model[b] + 
                        embedding_model[c])
        
        # Eng yaqin so'zni topish
        predicted_d = find_nearest_word(embedding_model, result_vector)
        
        if predicted_d == expected_d:
            correct += 1
    
    return correct / len(analogies)
```

### **5.2 Extrinsic Evaluation**

#### **Downstream Task Performance**
```python
# RAG pipeline da real performance
def evaluate_rag_embeddings(embedding_model, test_questions):
    results = []
    
    for question, expected_answer in test_questions:
        # Retrieval with embedding
        retrieved_docs = retrieve_with_embedding(
            question, 
            embedding_model
        )
        
        # Generate answer
        generated_answer = generate_answer(question, retrieved_docs)
        
        # Evaluate
        score = evaluate_answer(generated_answer, expected_answer)
        results.append(score)
    
    return np.mean(results)
```

## 6. Optimization va Best Practices

### **6.1 Embedding Dimension Optimization**
```python
# Dimension vs Performance trade-off
dimensions = [128, 256, 512, 768, 1024]
performance_results = []

for dim in dimensions:
    model = train_embedding_model(dimension=dim)
    performance = evaluate_model(model)
    memory_usage = calculate_memory(model)
    
    performance_results.append({
        'dimension': dim,
        'performance': performance,
        'memory': memory_usage,
        'speed': measure_inference_speed(model)
    })
```

### **6.2 Batch Processing Optimization**
```python
def optimize_embedding_generation():
    # GPU memory ni optimal ishlatish
    batch_size = calculate_optimal_batch_size()
    
    # Parallel processing
    def process_batch(texts_batch):
        with torch.no_grad():
            embeddings = model.encode(
                texts_batch,
                batch_size=batch_size,
                convert_to_tensor=True,
                device='cuda'
            )
        return embeddings.cpu().numpy()
    
    # Multi-processing
    from multiprocessing import Pool
    with Pool(processes=4) as pool:
        results = pool.map(process_batch, text_batches)
```

## 7. Multilingual va Cross-lingual Considerations

### **7.1 O'zbek tili uchun optimizatsiya**
```python
# O'zbek tiliga moslashgan embeddings
def adapt_for_uzbek():
    # 1. Uzbek corpus bilan fine-tuning
    uzbek_corpus = load_uzbek_texts()
    
    # 2. Morphological preprocessing
    def preprocess_uzbek_text(text):
        # O'zbek tilining morfologik xususiyatlari
        text = handle_uzbek_morphology(text)
        text = normalize_uzbek_script(text)  # Latin/Cyrillic
        return text
    
    # 3. Custom tokenization
    uzbek_tokenizer = create_uzbek_tokenizer()
    
    return adapted_model
```

### **7.2 Cross-lingual Retrieval**
```python
# Inglizcha query, o'zbekcha hujjat
def cross_lingual_retrieval(english_query, uzbek_documents):
    # Multilingual embedding model ishlatish
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    
    # Query embedding
    query_embedding = model.encode(english_query)
    
    # Document embeddings (cached)
    doc_embeddings = model.encode(uzbek_documents)
    
    # Cross-lingual similarity
    similarities = cosine_similarity([query_embedding], doc_embeddings)
    
    return similarities[0]
```

## Dars xulosasi

Embedding va vektor bazalar RAG tizimining yuragini tashkil etadi. To'g'ri model va index tanlash tizim samaradorligini sezilarli darajada oshiradi.

**Asosiy xulosalar:**
- Embedding modeli task ga mos tanlanishi kerak
- Vektor bazasi ma'lumot hajmi va performance talablariga qarab tanlanadi
- Multilingual support muhim ahamiyatga ega
- Regular evaluation va optimization zarur

## Keyingi darsga tayyorgarlik

Keyingi darsda text chunking va document preprocessing strategiyalarini o'rganamiz.

## Amaliy vazifa

1. Qaysi embedding model o'zbek tili uchun eng mos keladi va nima uchun?
2. 1 million hujjat uchun qaysi vektor bazasini tanlaysiz?
3. Cross-lingual qidiruv qachon kerak bo'ladi?
4. Embedding dimension qanday tanlanadi?