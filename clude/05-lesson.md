# RAG Mastery Course - 5-dars
## Retrieval Algoritmlar va Semantic Search

### 📚 Dars maqsadi
Bu darsda siz retrieval algoritmlarining turli xillari, semantic search prinsiplari va eng samarali qidiruv texnikalarini o'rganasiz.

---

## 1. Retrieval Algoritmlarining Turlari

### 1.1 Sparse Retrieval (Siyrak Qidiruv)
```python
# TF-IDF (Term Frequency-Inverse Document Frequency)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class TFIDFRetriever:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.document_vectors = None
        self.documents = []
    
    def fit(self, documents):
        """Document kolleksiyasini indexlash"""
        self.documents = documents
        self.document_vectors = self.vectorizer.fit_transform(documents)
        return self
    
    def search(self, query, top_k=5):
        """Query bo'yicha eng yaqin documentlarni topish"""
        query_vector = self.vectorizer.transform([query])
        
        # Cosine similarity hisoblash
        similarities = cosine_similarity(query_vector, self.document_vectors).flatten()
        
        # Top-k documentlarni olish
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                'document': self.documents[idx],
                'score': similarities[idx],
                'index': idx
            })
        
        return results

# BM25 Algorithm
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize

class BM25Retriever:
    def __init__(self, k1=1.2, b=0.75):
        self.k1 = k1  # Term frequency saturation parameter
        self.b = b    # Length normalization parameter
        self.bm25 = None
        self.documents = []
        self.tokenized_docs = []
    
    def preprocess_text(self, text):
        """Text preprocessing"""
        tokens = word_tokenize(text.lower())
        return [token for token in tokens if token.isalnum()]
    
    def fit(self, documents):
        """BM25 modelini tayyorlash"""
        self.documents = documents
        self.tokenized_docs = [self.preprocess_text(doc) for doc in documents]
        self.bm25 = BM25Okapi(self.tokenized_docs)
        return self
    
    def search(self, query, top_k=5):
        """Query bo'yicha qidiruv"""
        tokenized_query = self.preprocess_text(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        # Top-k natijalarni olish
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                'document': self.documents[idx],
                'score': scores[idx],
                'index': idx
            })
        
        return results
```

### 1.2 Dense Retrieval (Zich Qidiruv)
```python
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class DenseRetriever:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.documents = []
        self.embeddings = None
    
    def fit(self, documents):
        """Document embeddinglarini yaratish va indexlash"""
        self.documents = documents
        
        # Embeddinglar yaratish
        print("Embeddinglar yaratilmoqda...")
        self.embeddings = self.model.encode(documents, show_progress_bar=True)
        
        # FAISS index yaratish
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner Product (cosine similarity)
        
        # Embeddinglarni normalize qilish (cosine similarity uchun)
        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings)
        
        return self
    
    def search(self, query, top_k=5):
        """Semantic search"""
        # Query embeddingini yaratish
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Qidiruv
        scores, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            results.append({
                'document': self.documents[idx],
                'score': float(score),
                'index': int(idx),
                'rank': i + 1
            })
        
        return results
```

---

## 2. Hybrid Retrieval Strategies

### 2.1 Sparse + Dense Fusion
```python
class HybridRetriever:
    def __init__(self, sparse_weight=0.3, dense_weight=0.7):
        self.sparse_retriever = BM25Retriever()
        self.dense_retriever = DenseRetriever()
        self.sparse_weight = sparse_weight
        self.dense_weight = dense_weight
    
    def fit(self, documents):
        """Ikkala retrieverni ham tayyorlash"""
        print("Sparse retriever tayyorlanmoqda...")
        self.sparse_retriever.fit(documents)
        
        print("Dense retriever tayyorlanmoqda...")
        self.dense_retriever.fit(documents)
        
        return self
    
    def search(self, query, top_k=10):
        """Hybrid qidiruv: sparse va dense natijalarni birlashtirish"""
        # Har ikkala usuldan natijalar olish
        sparse_results = self.sparse_retriever.search(query, top_k=top_k*2)
        dense_results = self.dense_retriever.search(query, top_k=top_k*2)
        
        # Natijalarni birlashtirib, weighted score hisoblash
        combined_scores = {}
        
        # Sparse results
        for result in sparse_results:
            idx = result['index']
            normalized_score = result['score'] / max([r['score'] for r in sparse_results])
            combined_scores[idx] = self.sparse_weight * normalized_score
        
        # Dense results
        for result in dense_results:
            idx = result['index']
            if idx in combined_scores:
                combined_scores[idx] += self.dense_weight * result['score']
            else:
                combined_scores[idx] = self.dense_weight * result['score']
        
        # Top-k natijalarni saralash
        sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        final_results = []
        for idx, score in sorted_results[:top_k]:
            final_results.append({
                'document': self.sparse_retriever.documents[idx],
                'score': score,
                'index': idx
            })
        
        return final_results
```

### 2.2 Reciprocal Rank Fusion (RRF)
```python
class RRFRetriever:
    def __init__(self, retrievers, k=60):
        self.retrievers = retrievers  # Retrieverlar listi
        self.k = k  # RRF parameter
    
    def search(self, query, top_k=10):
        """Reciprocal Rank Fusion algoritmini qo'llash"""
        all_results = {}
        
        # Har bir retrieverdan natijalar olish
        for retriever in self.retrievers:
            results = retriever.search(query, top_k=top_k*2)
            
            for rank, result in enumerate(results, 1):
                doc_id = result['index']
                rrf_score = 1 / (self.k + rank)
                
                if doc_id in all_results:
                    all_results[doc_id]['rrf_score'] += rrf_score
                else:
                    all_results[doc_id] = {
                        'document': result['document'],
                        'rrf_score': rrf_score,
                        'index': doc_id
                    }
        
        # RRF scorega ko'ra saralash
        sorted_results = sorted(all_results.values(), 
                              key=lambda x: x['rrf_score'], reverse=True)
        
        return sorted_results[:top_k]
```

---

## 3. Advanced Retrieval Techniques

### 3.1 Query Expansion
```python
from transformers import pipeline
import synonyms

class QueryExpander:
    def __init__(self):
        # Pre-trained model for query expansion
        self.generator = pipeline("text2text-generation", 
                                model="t5-small")
    
    def expand_with_synonyms(self, query):
        """Sinonimlar bilan query kengaytirish"""
        words = query.split()
        expanded_terms = []
        
        for word in words:
            expanded_terms.append(word)
            # Synonyms library dan foydalanish
            syns = synonyms.nearby(word)
            if syns:
                expanded_terms.extend(syns[:2])  # Eng yaqin 2ta sinonim
        
        return " ".join(expanded_terms)
    
    def expand_with_generation(self, query):
        """LLM yordamida query kengaytirish"""
        prompt = f"Expand this search query with related terms: {query}"
        expanded = self.generator(prompt, max_length=50, num_return_sequences=1)
        return expanded[0]['generated_text']
    
    def multi_query_generation(self, query, num_queries=3):
        """Bir querydan bir nechta variant yaratish"""
        queries = [query]  # Original query
        
        for i in range(num_queries - 1):
            prompt = f"Rephrase this search query: {query}"
            rephrased = self.generator(prompt, max_length=30, 
                                     num_return_sequences=1)
            queries.append(rephrased[0]['generated_text'])
        
        return queries

# Query expansion bilan retriever
class ExpandedRetriever:
    def __init__(self, base_retriever):
        self.base_retriever = base_retriever
        self.expander = QueryExpander()
    
    def search(self, query, top_k=10, use_expansion=True):
        if not use_expansion:
            return self.base_retriever.search(query, top_k)
        
        # Query expansion
        expanded_query = self.expander.expand_with_synonyms(query)
        multi_queries = self.expander.multi_query_generation(query, 3)
        
        # Har bir query variant uchun qidiruv
        all_results = {}
        
        for q in [query, expanded_query] + multi_queries:
            results = self.base_retriever.search(q, top_k=top_k)
            
            for result in results:
                doc_id = result['index']
                if doc_id in all_results:
                    all_results[doc_id]['score'] += result['score']
                    all_results[doc_id]['count'] += 1
                else:
                    all_results[doc_id] = {
                        'document': result['document'],
                        'score': result['score'],
                        'count': 1,
                        'index': doc_id
                    }
        
        # Average score bo'yicha saralash
        for doc_id in all_results:
            all_results[doc_id]['avg_score'] = (
                all_results[doc_id]['score'] / all_results[doc_id]['count']
            )
        
        sorted_results = sorted(all_results.values(), 
                              key=lambda x: x['avg_score'], reverse=True)
        
        return sorted_results[:top_k]
```

### 3.2 Re-ranking
```python
from sentence_transformers import CrossEncoder

class ReRanker:
    def __init__(self, model_name='cross-encoder/ms-marco-MiniLM-L-6-v2'):
        self.cross_encoder = CrossEncoder(model_name)
    
    def rerank(self, query, candidates, top_k=5):
        """Cross-encoder yordamida re-ranking"""
        if len(candidates) <= top_k:
            return candidates
        
        # Query-document juftliklarini tayyorlash
        pairs = []
        for candidate in candidates:
            pairs.append([query, candidate['document'][:512]])  # Text uzunligini cheklash
        
        # Cross-encoder score hisoblash
        scores = self.cross_encoder.predict(pairs)
        
        # Scorelarni candidatelarga qo'shish
        for i, candidate in enumerate(candidates):
            candidate['rerank_score'] = scores[i]
        
        # Re-ranking
        reranked = sorted(candidates, key=lambda x: x['rerank_score'], reverse=True)
        
        return reranked[:top_k]

# Complete pipeline with re-ranking
class AdvancedRetriever:
    def __init__(self, base_retriever, use_reranking=True):
        self.base_retriever = base_retriever
        self.use_reranking = use_reranking
        if use_reranking:
            self.reranker = ReRanker()
    
    def search(self, query, top_k=5, retrieval_k=20):
        # Avval ko'proq natija olish
        initial_results = self.base_retriever.search(query, top_k=retrieval_k)
        
        if not self.use_reranking or len(initial_results) <= top_k:
            return initial_results[:top_k]
        
        # Re-ranking qo'llash
        reranked_results = self.reranker.rerank(query, initial_results, top_k)
        
        return reranked_results
```

---

## 4. Contextual Retrieval

### 4.1 Conversational Retrieval
```python
class ConversationalRetriever:
    def __init__(self, base_retriever):
        self.base_retriever = base_retriever
        self.conversation_history = []
    
    def add_turn(self, query, response):
        """Suhbat tarixiga navbat qo'shish"""
        self.conversation_history.append({
            'query': query,
            'response': response
        })
    
    def contextualize_query(self, current_query):
        """Joriy queryni kontekst bilan boyitish"""
        if not self.conversation_history:
            return current_query
        
        # Oxirgi bir necha suhbat navbatini olish
        recent_context = self.conversation_history[-3:]  # Oxirgi 3ta
        
        context_str = ""
        for turn in recent_context:
            context_str += f"Q: {turn['query']}\nA: {turn['response'][:200]}...\n"
        
        contextualized_query = f"""
        Suhbat konteksti:
        {context_str}
        
        Joriy savol: {current_query}
        
        Kontekstni hisobga olib, quyidagi savolga javob berish uchun kerakli ma'lumotni qidiring.
        """
        
        return contextualized_query
    
    def search(self, query, top_k=5):
        """Kontekstli qidiruv"""
        contextualized_query = self.contextualize_query(query)
        results = self.base_retriever.search(contextualized_query, top_k)
        return results
```

### 4.2 Multi-modal Retrieval
```python
import clip
import torch
from PIL import Image

class MultiModalRetriever:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.text_embeddings = None
        self.image_embeddings = None
        self.documents = []
        self.images = []
    
    def encode_texts(self, texts):
        """Textlarni encode qilish"""
        text_tokens = clip.tokenize(texts).to(self.device)
        with torch.no_grad():
            text_embeddings = self.model.encode_text(text_tokens)
        return text_embeddings
    
    def encode_images(self, image_paths):
        """Rasmlarni encode qilish"""
        images = []
        for img_path in image_paths:
            image = self.preprocess(Image.open(img_path)).unsqueeze(0).to(self.device)
            images.append(image)
        
        images_tensor = torch.cat(images)
        with torch.no_grad():
            image_embeddings = self.model.encode_image(images_tensor)
        return image_embeddings
    
    def fit(self, documents, image_paths=None):
        """Multi-modal contentni indexlash"""
        self.documents = documents
        self.text_embeddings = self.encode_texts(documents)
        
        if image_paths:
            self.images = image_paths
            self.image_embeddings = self.encode_images(image_paths)
    
    def search_text(self, query, top_k=5):
        """Text orqali qidiruv"""
        query_embedding = self.encode_texts([query])
        
        # Text bilan similarity
        text_similarities = torch.cosine_similarity(query_embedding, self.text_embeddings)
        
        results = []
        if self.image_embeddings is not None:
            # Image bilan ham similarity
            image_similarities = torch.cosine_similarity(query_embedding, self.image_embeddings)
            
            # Combined results
            for i, (text_sim, img_sim) in enumerate(zip(text_similarities, image_similarities)):
                combined_score = 0.7 * text_sim + 0.3 * img_sim  # Weighted combination
                results.append({
                    'document': self.documents[i],
                    'image': self.images[i] if i < len(self.images) else None,
                    'score': combined_score.item(),
                    'text_score': text_sim.item(),
                    'image_score': img_sim.item()
                })
        else:
            for i, sim in enumerate(text_similarities):
                results.append({
                    'document': self.documents[i],
                    'score': sim.item()
                })
        
        # Sort by score
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]
```

---

## 5. Retrieval Evaluation

### 5.1 Evaluation Metrics
```python
import numpy as np
from collections import defaultdict

class RetrievalEvaluator:
    def __init__(self):
        pass
    
    def precision_at_k(self, retrieved_docs, relevant_docs, k):
        """Precision@K hisoblash"""
        retrieved_k = retrieved_docs[:k]
        relevant_retrieved = len(set(retrieved_k) & set(relevant_docs))
        return relevant_retrieved / k if k > 0 else 0
    
    def recall_at_k(self, retrieved_docs, relevant_docs, k):
        """Recall@K hisoblash"""
        retrieved_k = retrieved_docs[:k]
        relevant_retrieved = len(set(retrieved_k) & set(relevant_docs))
        return relevant_retrieved / len(relevant_docs) if len(relevant_docs) > 0 else 0
    
    def average_precision(self, retrieved_docs, relevant_docs):
        """Average Precision hisoblash"""
        if not relevant_docs:
            return 0
        
        precisions = []
        relevant_count = 0
        
        for i, doc in enumerate(retrieved_docs):
            if doc in relevant_docs:
                relevant_count += 1
                precision = relevant_count / (i + 1)
                precisions.append(precision)
        
        return np.mean(precisions) if precisions else 0
    
    def mean_average_precision(self, all_retrieved, all_relevant):
        """Mean Average Precision hisoblash"""
        aps = []
        for retrieved_docs, relevant_docs in zip(all_retrieved, all_relevant):
            ap = self.average_precision(retrieved_docs, relevant_docs)
            aps.append(ap)
        
        return np.mean(aps)
    
    def ndcg_at_k(self, retrieved_docs, relevance_scores, k):
        """Normalized Discounted Cumulative Gain@K"""
        def dcg_at_k(scores, k):
            scores = np.array(scores)[:k]
            return np.sum(scores / np.log2(np.arange(2, len(scores) + 2)))
        
        # Actual DCG
        actual_scores = [relevance_scores.get(doc, 0) for doc in retrieved_docs[:k]]
        actual_dcg = dcg_at_k(actual_scores, k)
        
        # Ideal DCG
        ideal_scores = sorted(relevance_scores.values(), reverse=True)
        ideal_dcg = dcg_at_k(ideal_scores, k)
        
        return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0
    
    def mrr(self, all_retrieved, all_relevant):
        """Mean Reciprocal Rank"""
        reciprocal_ranks = []
        
        for retrieved_docs, relevant_docs in zip(all_retrieved, all_relevant):
            rr = 0
            for i, doc in enumerate(retrieved_docs):
                if doc in relevant_docs:
                    rr = 1 / (i + 1)
                    break
            reciprocal_ranks.append(rr)
        
        return np.mean(reciprocal_ranks)

# Evaluation pipeline
def evaluate_retriever(retriever, test_queries, ground_truth):
    """Retrieverni to'liq baholash"""
    evaluator = RetrievalEvaluator()
    
    all_retrieved = []
    all_relevant = []
    
    for query, relevant_docs in zip(test_queries, ground_truth):
        # Retrieval
        results = retriever.search(query, top_k=20)
        retrieved_docs = [r['index'] for r in results]
        
        all_retrieved.append(retrieved_docs)
        all_relevant.append(relevant_docs)
    
    # Metrics hisoblash
    metrics = {}
    
    # Precision@K va Recall@K
    for k in [1, 3, 5, 10]:
        precisions = [evaluator.precision_at_k(ret, rel, k) 
                     for ret, rel in zip(all_retrieved, all_relevant)]
        recalls = [evaluator.recall_at_k(ret, rel, k) 
                  for ret, rel in zip(all_retrieved, all_relevant)]
        
        metrics[f'Precision@{k}'] = np.mean(precisions)
        metrics[f'Recall@{k}'] = np.mean(recalls)
    
    # MAP va MRR
    metrics['MAP'] = evaluator.mean_average_precision(all_retrieved, all_relevant)
    metrics['MRR'] = evaluator.mrr(all_retrieved, all_relevant)
    
    return metrics
```

---

## 6. Amaliy Mashq

```python
# To'liq retrieval pipeline
class ProductionRetriever:
    def __init__(self, config):
        self.config = config
        self.retrievers = self._build_retrievers()
        self.reranker = ReRanker() if config.get('use_reranking') else None
        self.query_expander = QueryExpander() if config.get('use_expansion') else None
    
    def _build_retrievers(self):
        """Konfiguratsiyaga asoslangan retrieverlar yaratish"""
        retrievers = []
        
        if self.config.get('use_bm25'):
            retrievers.append(BM25Retriever())
        
        if self.config.get('use_dense'):
            model_name = self.config.get('dense_model', 'all-MiniLM-L6-v2')
            retrievers.append(DenseRetriever(model_name))
        
        return retrievers
    
    def fit(self, documents):
        """Barcha retrieverlani tayyorlash"""
        for retriever in self.retrievers:
            retriever.fit(documents)
        return self
    
    def search(self, query, top_k=5):
        """Advanced qidiruv pipeline"""
        # Query expansion
        if self.query_expander:
            expanded_queries = self.query_expander.multi_query_generation(query)
        else:
            expanded_queries = [query]
        
        # Multi-retriever search
        all_candidates = {}
        
        for exp_query in expanded_queries:
            if len(self.retrievers) == 1:
                # Single retriever
                results = self.retrievers[0].search(exp_query, top_k=top_k*3)
                for result in results:
                    doc_id = result['index']
                    if doc_id in all_candidates:
                        all_candidates[doc_id]['score'] += result['score']
                    else:
                        all_candidates[doc_id] = result
            else:
                # Hybrid approach
                hybrid_retriever = RRFRetriever(self.retrievers)
                results = hybrid_retriever.search(exp_query, top_k=top_k*3)
                for result in results:
                    doc_id = result['index']
                    if doc_id in all_candidates:
                        all_candidates[doc_id]['rrf_score'] += result['rrf_score']
                    else:
                        all_candidates[doc_id] = result
        
        # Candidatelarni saralash
        candidates = list(all_candidates.values())
        score_key = 'rrf_score' if len(self.retrievers) > 1 else 'score'
        candidates.sort(key=lambda x: x[score_key], reverse=True)
        
        # Re-ranking
        if self.reranker and len(candidates) > top_k:
            final_results = self.reranker.rerank(query, candidates[:top_k*2], top_k)
        else:
            final_results = candidates[:top_k]
        
        return final_results

# Configuration example
config = {
    'use_bm25': True,
    'use_dense': True,
    'dense_model': 'all-MiniLM-L6-v2',
    'use_reranking': True,
    'use_expansion': True
}

# Qo'llanish
retriever = ProductionRetriever(config)
# retriever.fit(documents)
# results = retriever.search("Sizning savolingiz", top_k=5)
```

---

## 📝 Amaliy Vazifa

1. **BM25 vs Dense Retrieval Taqqoslash:**
   - Ikki xil retrieverni bir xil dataset ustida sinab ko'ring
   - Har xil turdagi querylar uchun qaysi biri yaxshiroq ishlashini aniqlang

2. **Hybrid Retrieval Implementation:**
   - RRF va weighted fusion usullarini implement qiling
   - Optimal weight parametrlarini topish uchun eksperiment o'tkazing

3. **Query Expansion Eksperimenti:**
   - Synonym-based va generation-based expansionni taqqoslang
   - Query expansion qachon foydali va qachon zararli ekanligini aniqlang

4. **Evaluation Pipeline:**
   - O'zingizning test datasetingizni yarating
   - Turli retrieval strategiyalarini P@K, MAP, MRR metrikalari bilan baholang

---

## 🎯 Keyingi Dars: Generation va Prompt Engineering

Keyingi darsda biz RAG sistemasining generation qismini o'rganamiz - qanday qilib olingan documentlar asosida sifatli javoblar generatsiya qilish va prompt engineeringning eng yaxshi amaliyotlari.