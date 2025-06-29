# Chunking va Document Preprocessing

## Dars maqsadi
Ushbu darsda siz hujjatlarni RAG uchun tayyorlash, chunking strategiyalari va preprocessing texnikalarini chuqur o'rganasiz.

## 1. Document Preprocessing Fundamentals

### **1.1 Nima uchun preprocessing kerak?**

RAG tizimida muvaffaqiyat 80% to'g'ri preprocessing ga bog'liq:
- **Noise removal:** Keraksiz ma'lumotlarni olib tashlash
- **Structure preservation:** Muhim strukturani saqlash  
- **Consistency:** Bir xil formatga keltirish
- **Quality enhancement:** Ma'lumot sifatini oshirish

### **1.2 Document Types va ularning muammolari**

```python
# Har xil format muammolari
document_challenges = {
    'PDF': ['Scanned images', 'Complex layouts', 'Tables', 'Headers/footers'],
    'DOCX': ['Embedded objects', 'Comments', 'Track changes', 'Styles'],
    'HTML': ['Tags', 'CSS', 'JavaScript', 'Navigation elements'],
    'CSV': ['Headers', 'Empty cells', 'Data types', 'Encoding'],
    'JSON': ['Nested structure', 'Array handling', 'Schema variations']
}
```

## 2. Text Extraction va Cleaning

### **2.1 PDF Processing**

#### **Basic PDF extraction**
```python
import PyPDF2
import fitz  # PyMuPDF
from pdfplumber import PDF

def extract_from_pdf_basic(pdf_path):
    """Basic PDF text extraction"""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

def extract_from_pdf_advanced(pdf_path):
    """Advanced PDF extraction with layout preservation"""
    doc = fitz.open(pdf_path)
    full_text = ""
    
    for page_num in range(doc.page_count):
        page = doc[page_num]
        
        # Text extraction with position info
        blocks = page.get_text("dict")
        
        for block in blocks["blocks"]:
            if "lines" in block:
                for line in block["lines"]:
                    line_text = ""
                    for span in line["spans"]:
                        line_text += span["text"]
                    full_text += line_text + "\n"
    
    return full_text
```

#### **Table extraction from PDF**
```python
import tabula
import camelot

def extract_tables_from_pdf(pdf_path):
    """PDF dan jadvallarni ajratib olish"""
    
    # Tabula ishlatish
    tables_tabula = tabula.read_pdf(pdf_path, pages='all')
    
    # Camelot ishlatish (aniqroq)
    tables_camelot = camelot.read_pdf(pdf_path, pages='all')
    
    processed_tables = []
    for table in tables_camelot:
        # Jadval sifatini tekshirish
        if table.parsing_report['accuracy'] > 80:
            # CSV formatga o'tkazish
            csv_string = table.df.to_csv(index=False)
            processed_tables.append(csv_string)
    
    return processed_tables
```

### **2.2 DOCX Processing**

```python
import docx
from docx.document import Document
from docx.oxml.table import CT_Table
from docx.oxml.text.paragraph import CT_P
from docx.table import _Cell, Table
from docx.text.paragraph import Paragraph

def extract_from_docx_comprehensive(docx_path):
    """DOCX dan barcha elementlarni ajratib olish"""
    document = docx.Document(docx_path)
    
    full_text = ""
    
    def iter_block_items(parent):
        """DOCX elementlarini ketma-ket o'qish"""
        for child in parent.element.body:
            if isinstance(child, CT_P):
                paragraph = Paragraph(child, parent)
                full_text += paragraph.text + "\n"
            elif isinstance(child, CT_Table):
                table = Table(child, parent)
                table_text = extract_table_text(table)
                full_text += table_text + "\n"
    
    def extract_table_text(table):
        """Jadval matnini ajratib olish"""
        table_data = []
        for row in table.rows:
            row_data = []
            for cell in row.cells:
                row_data.append(cell.text.strip())
            table_data.append(" | ".join(row_data))
        return "\n".join(table_data)
    
    iter_block_items(document)
    return full_text
```

### **2.3 HTML Processing**

```python
from bs4 import BeautifulSoup
import html2text
import re

def extract_from_html_smart(html_content):
    """HTML dan semantik matnni ajratib olish"""
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Navigation va ads elementlarini olib tashlash
    for element in soup(['nav', 'footer', 'header', 'aside', 'script', 'style']):
        element.decompose()
    
    # Ads klaslaridagi elementlar
    ads_classes = ['advertisement', 'ad-', 'google-ad', 'sidebar']
    for class_name in ads_classes:
        for element in soup.find_all(class_=re.compile(class_name)):
            element.decompose()
    
    # Asosiy kontent topish
    main_content = (
        soup.find('main') or 
        soup.find('article') or 
        soup.find('div', class_=re.compile('content|main|article'))
    )
    
    if main_content:
        # HTML2Text ishlatish
        h = html2text.HTML2Text()
        h.ignore_links = True
        h.ignore_images = True
        text = h.handle(str(main_content))
    else:
        text = soup.get_text()
    
    # Tozalash
    text = re.sub(r'\n\s*\n', '\n\n', text)  # Ko'p bo'sh qatorlarni olib tashlash
    text = re.sub(r'[ \t]+', ' ', text)      # Ko'p probel/tab larni olib tashlash
    
    return text.strip()
```

## 3. Text Normalization va Cleaning

### **3.1 Basic Text Cleaning**

```python
import re
import unicodedata

def basic_text_cleaning(text):
    """Asosiy matn tozalash"""
    
    # Unicode normalization
    text = unicodedata.normalize('NFKD', text)
    
    # HTML entities ni decode qilish
    import html
    text = html.unescape(text)
    
    # URL larni olib tashlash
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    # Email larni olib tashlash
    text = re.sub(r'\S+@\S+', '', text)
    
    # Ko'p punktuatsiya belgisini tozalash
    text = re.sub(r'[.]{3,}', '...', text)
    text = re.sub(r'[!]{2,}', '!', text)
    text = re.sub(r'[?]{2,}', '?', text)
    
    # Ko'p bo'shliqlarni tozalash
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def advanced_text_cleaning(text, language='uzbek'):
    """Chuqurroq matn tozalash"""
    
    # Language-specific cleaning
    if language == 'uzbek':
        # O'zbek tilida Latin-Kirill konvertatsiya
        text = normalize_uzbek_script(text)
        
        # O'zbekcha stop words yaqinidagi noise
        uzbek_noise_patterns = [
            r'\b(elon|reklama|advertisement)\b',
            r'\d{2}[.-]\d{2}[.-]\d{4}',  # Date patterns
            r'\+\d{3}\s?\d{2}\s?\d{3}\s?\d{2}\s?\d{2}',  # Phone numbers
        ]
        
        for pattern in uzbek_noise_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # Scientific notation tozalash
    text = re.sub(r'\d+\.?\d*e[+-]?\d+', '[SCIENTIFIC_NUMBER]', text)
    
    # References tozalash
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'\(\d+\)', '', text)
    
    return text

def normalize_uzbek_script(text):
    """O'zbek matnini normalizatsiya qilish"""
    # Kirill-Latin konversiya jadvali
    cyrillic_to_latin = {
        'а': 'a', 'б': 'b', 'в': 'v', 'г': 'g', 'д': 'd',
        'е': 'e', 'ё': 'yo', 'ж': 'j', 'з': 'z', 'и': 'i',
        'й': 'y', 'к': 'k', 'л': 'l', 'м': 'm', 'н': 'n',
        'о': 'o', 'п': 'p', 'р': 'r', 'с': 's', 'т': 't',
        'у': 'u', 'ф': 'f', 'х': 'x', 'ц': 'ts', 'ч': 'ch',
        'ш': 'sh', 'щ': 'shch', 'ъ': '', 'ы': 'i', 'ь': '',
        'э': 'e', 'ю': 'yu', 'я': 'ya', 'ў': 'oʻ', 'қ': 'q',
        'ғ': 'gʻ', 'ҳ': 'h'
    }
    
    # Konversiya
    for cyrillic, latin in cyrillic_to_latin.items():
        text = text.replace(cyrillic, latin)
        text = text.replace(cyrillic.upper(), latin.capitalize())
    
    return text
```

### **3.2 Structure Preservation**

```python
def preserve_document_structure(text):
    """Hujjat strukturasini saqlash"""
    
    # Headings ni belgilash
    text = re.sub(r'^#{1,6}\s+(.+)$', r'[HEADING] \1', text, flags=re.MULTILINE)
    
    # Lists ni belgilash
    text = re.sub(r'^\s*[-*+]\s+(.+)$', r'[LIST_ITEM] \1', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s+(.+)$', r'[NUMBERED_ITEM] \1', text, flags=re.MULTILINE)
    
    # Code blocks
    text = re.sub(r'```(.+?)```', r'[CODE_BLOCK] \1 [/CODE_BLOCK]', text, flags=re.DOTALL)
    
    # Tables
    text = re.sub(r'\|(.+?)\|', r'[TABLE_ROW] \1 [/TABLE_ROW]', text)
    
    # Quotes
    text = re.sub(r'^>\s+(.+)$', r'[QUOTE] \1', text, flags=re.MULTILINE)
    
    return text
```

## 4. Chunking Strategies

### **4.1 Fixed-size Chunking**

```python
def fixed_size_chunking(text, chunk_size=500, overlap=50):
    """Belgilangan o'lchamda bo'lish"""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk_words = words[i:i + chunk_size]
        chunk_text = ' '.join(chunk_words)
        
        chunks.append({
            'text': chunk_text,
            'start_index': i,
            'end_index': min(i + chunk_size, len(words)),
            'word_count': len(chunk_words)
        })
    
    return chunks

def token_based_chunking(text, max_tokens=512, model_name='bert-base-uncased'):
    """Token asosida bo'lish"""
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Text ni tokenize qilish
    tokens = tokenizer.encode(text, add_special_tokens=False)
    
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i:i + max_tokens]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        
        chunks.append({
            'text': chunk_text,
            'tokens': chunk_tokens,
            'token_count': len(chunk_tokens)
        })
    
    return chunks
```

### **4.2 Semantic Chunking**

```python
import spacy
from sentence_transformers import SentenceTransformer

def semantic_chunking(text, similarity_threshold=0.5, max_chunk_size=1000):
    """Ma'no bo'yicha bo'lish"""
    
    # Sentences ga bo'lish
    nlp = spacy.load('en_core_web_sm')  # yoki uzbek model
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]
    
    # Sentence embeddings
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    embeddings = model.encode(sentences)
    
    # Similarity matrix
    from sklearn.metrics.pairwise import cosine_similarity
    similarity_matrix = cosine_similarity(embeddings)
    
    chunks = []
    current_chunk = []
    current_chunk_size = 0
    
    for i, sentence in enumerate(sentences):
        sentence_size = len(sentence.split())
        
        if not current_chunk:
            current_chunk.append(sentence)
            current_chunk_size = sentence_size
            continue
        
        # Oldingi jumla bilan o'xshashlikni tekshirish
        similarity = similarity_matrix[i][i-1]
        
        if (similarity > similarity_threshold and 
            current_chunk_size + sentence_size <= max_chunk_size):
            current_chunk.append(sentence)
            current_chunk_size += sentence_size
        else:
            # Yangi chunk boshlash
            chunks.append({
                'text': ' '.join(current_chunk),
                'sentence_count': len(current_chunk),
                'word_count': current_chunk_size
            })
            current_chunk = [sentence]
            current_chunk_size = sentence_size
    
    # Oxirgi chunk qo'shish
    if current_chunk:
        chunks.append({
            'text': ' '.join(current_chunk),
            'sentence_count': len(current_chunk),
            'word_count': current_chunk_size
        })
    
    return chunks
```

### **4.3 Hierarchical Chunking**

```python
def hierarchical_chunking(text, levels=[100, 500, 2000]):
    """Ierarxik bo'lish"""
    
    # Document -> Sections -> Paragraphs -> Sentences
    
    def create_hierarchy(text, chunk_sizes):
        hierarchy = {'text': text, 'children': []}
        
        if len(chunk_sizes) == 1:
            # Base case: final chunking
            chunks = fixed_size_chunking(text, chunk_sizes[0])
            hierarchy['children'] = [{'text': chunk['text'], 'children': []} 
                                   for chunk in chunks]
        else:
            # Recursive case
            current_size = chunk_sizes[0]
            remaining_sizes = chunk_sizes[1:]
            
            chunks = fixed_size_chunking(text, current_size)
            for chunk in chunks:
                child_hierarchy = create_hierarchy(chunk['text'], remaining_sizes)
                hierarchy['children'].append(child_hierarchy)
        
        return hierarchy
    
    return create_hierarchy(text, levels)

def extract_chunks_from_hierarchy(hierarchy, target_level=1):
    """Ierarxiyadan ma'lum darajadagi chunklarni olish"""
    chunks = []
    
    def traverse(node, current_level=0):
        if current_level == target_level:
            chunks.append(node['text'])
        else:
            for child in node['children']:
                traverse(child, current_level + 1)
    
    traverse(hierarchy)
    return chunks
```

### **4.4 Document Structure-aware Chunking**

```python
def structure_aware_chunking(text):
    """Hujjat strukturasiga asoslangan bo'lish"""
    
    lines = text.split('\n')
    chunks = []
    current_chunk = []
    current_section = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Heading detection
        if re.match(r'^#{1,6}\s+', line) or re.match(r'^[A-Z][A-Z\s]+$', line):
            # Oldingi section ni yakunlash
            if current_chunk:
                chunks.append({
                    'text': '\n'.join(current_chunk),
                    'section': current_section,
                    'type': 'section'
                })
            
            # Yangi section boshlash
            current_section = line
            current_chunk = [line]
        
        # Table detection
        elif '|' in line and line.count('|') >= 2:
            if current_chunk and current_chunk[-1] != '[TABLE_START]':
                current_chunk.append('[TABLE_START]')
            current_chunk.append(line)
        
        # List detection
        elif re.match(r'^\s*[-*+]\s+', line) or re.match(r'^\s*\d+\.\s+', line):
            current_chunk.append(line)
        
        # Regular paragraph
        else:
            current_chunk.append(line)
            
            # Paragraph chunking
            if len(' '.join(current_chunk).split()) > 300:
                chunks.append({
                    'text': '\n'.join(current_chunk),
                    'section': current_section,
                    'type': 'paragraph'
                })
                current_chunk = []
    
    # Oxirgi chunk
    if current_chunk:
        chunks.append({
            'text': '\n'.join(current_chunk),
            'section': current_section,
            'type': 'final'
        })
    
    return chunks
```

## 5. Advanced Chunking Techniques

### **5.1 Context-preserving Chunking**

```python
def context_preserving_chunking(text, context_size=100, chunk_size=500):
    """Kontekstni saqlagan holda bo'lish"""
    
    words = text.split()
    chunks_with_context = []
    
    for i in range(0, len(words), chunk_size):
        # Asosiy chunk
        main_chunk = words[i:i + chunk_size]
        
        # Oldingi kontekst
        prev_context_start = max(0, i - context_size)
        prev_context = words[prev_context_start:i] if i > 0 else []
        
        # Keyingi kontekst
        next_context_end = min(len(words), i + chunk_size + context_size)
        next_context = words[i + chunk_size:next_context_end] if i + chunk_size < len(words) else []
        
        chunk_with_context = {
            'main_text': ' '.join(main_chunk),
            'prev_context': ' '.join(prev_context),
            'next_context': ' '.join(next_context),
            'full_text': ' '.join(prev_context + main_chunk + next_context),
            'start_index': i,
            'end_index': i + len(main_chunk)
        }
        
        chunks_with_context.append(chunk_with_context)
    
    return chunks_with_context
```

### **5.2 Topic-based Chunking**

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

def topic_based_chunking(text, n_topics=5, min_chunk_size=200):
    """Mavzu bo'yicha bo'lish"""
    
    # Sentences ga bo'lish
    sentences = text.split('.')
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    
    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
    sentence_vectors = vectorizer.fit_transform(sentences)
    
    # Clustering
    kmeans = KMeans(n_clusters=n_topics, random_state=42)
    clusters = kmeans.fit_predict(sentence_vectors)
    
    # Cluster bo'yicha chunking
    topic_chunks = {}
    for i, cluster_id in enumerate(clusters):
        if cluster_id not in topic_chunks:
            topic_chunks[cluster_id] = []
        topic_chunks[cluster_id].append(sentences[i])
    
    # Kichik chunklarni birlashtirish
    final_chunks = []
    for topic_id, topic_sentences in topic_chunks.items():
        topic_text = '. '.join(topic_sentences)
        
        if len(topic_text.split()) < min_chunk_size:
            # Kichik topic ni eng yaqin topic bilan birlashtirish
            pass  # Implementation details
        else:
            final_chunks.append({
                'text': topic_text,
                'topic_id': topic_id,
                'sentence_count': len(topic_sentences)
            })
    
    return final_chunks
```

## 6. Chunking Quality Evaluation

### **6.1 Chunk Quality Metrics**

```python
def evaluate_chunk_quality(chunks):
    """Chunk sifatini baholash"""
    
    metrics = {
        'total_chunks': len(chunks),
        'avg_chunk_size': 0,
        'size_variance': 0,
        'coverage_ratio': 0,
        'overlap_ratio': 0,
        'semantic_coherence': 0
    }
    
    # Size metrics
    chunk_sizes = [len(chunk['text'].split()) for chunk in chunks]
    metrics['avg_chunk_size'] = sum(chunk_sizes) / len(chunk_sizes)
    metrics['size_variance'] = np.var(chunk_sizes)
    
    # Coverage ratio
    total_words = sum(chunk_sizes)
    original_words = len(' '.join([chunk['text'] for chunk in chunks]).split())
    metrics['coverage_ratio'] = total_words / original_words
    
    # Semantic coherence (sentence transformer orqali)
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    
    coherence_scores = []
    for chunk in chunks:
        sentences = chunk['text'].split('.')
        if len(sentences) > 1:
            sentence_embeddings = model.encode(sentences)
            avg_similarity = np.mean(cosine_similarity(sentence_embeddings))
            coherence_scores.append(avg_similarity)
    
    metrics['semantic_coherence'] = np.mean(coherence_scores) if coherence_scores else 0
    
    return metrics
```

### **6.2 Optimal Chunking Strategy Selection**

```python
def find_optimal_chunking_strategy(text, evaluation_queries=None):
    """Optimal chunking strategiyasini topish"""
    
    strategies = [
        ('fixed_size', lambda t: fixed_size_chunking(t, 500, 50)),
        ('semantic', lambda t: semantic_chunking(t, 0.5, 1000)),
        ('structure_aware', lambda t: structure_aware_chunking(t)),
        ('token_based', lambda t: token_based_chunking(t, 512))
    ]
    
    results = {}
    
    for strategy_name, strategy_func in strategies:
        chunks = strategy_func(text)
        quality_metrics = evaluate_chunk_quality(chunks)
        
        # Evaluation queries bilan test qilish
        if evaluation_queries:
            retrieval_performance = evaluate_retrieval_performance(
                chunks, evaluation_queries
            )
            quality_metrics.update(retrieval_performance)
        
        results[strategy_name] = quality_metrics
    
    # Eng yaxshi strategiyani tanlash
    best_strategy = max(results.items(), 
                       key=lambda x: x[1]['semantic_coherence'])
    
    return best_strategy

def evaluate_retrieval_performance(chunks, queries):
    """Retrieval performance ni baholash"""
    # Implementation: chunks dan query uchun relevant ma'lumot topish
    # Precision@K, Recall@K hisoblash
    pass
```

## 7. Production Best Practices

### **7.1 Scalable Chunking Pipeline**

```python
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor

class ChunkingPipeline:
    def __init__(self, strategy='semantic', parallel=True):
        self.strategy = strategy
        self.parallel = parallel
        
    def process_documents(self, documents):
        """Ko'p hujjatni parallel processing"""
        
        if self.parallel:
            with ThreadPoolExecutor(max_workers=mp.cpu_count()) as executor:
                futures = [executor.submit(self.process_single_document, doc) 
                          for doc in documents]
                results = [future.result() for future in futures]
        else:
            results = [self.process_single_document(doc) for doc in documents]
        
        return results
    
    def process_single_document(self, document):
        """Bitta hujjatni processing qilish"""
        
        # 1. Text extraction
        if document['type'] == 'pdf':
            text = extract_from_pdf_advanced(document['path'])
        elif document['type'] == 'docx':
            text = extract_from_docx_comprehensive(document['path'])
        else:
            text = document['content']
        
        # 2. Cleaning
        clean_text = advanced_text_cleaning(text)
        
        # 3. Chunking
        if self.strategy == 'semantic':
            chunks = semantic_chunking(clean_text)
        elif self.strategy == 'structure_aware':
            chunks = structure_aware_chunking(clean_text)
        else:
            chunks = fixed_size_chunking(clean_text)
        
        # 4. Metadata addition
        for i, chunk in enumerate(chunks):
            chunk['document_id'] = document['id']
            chunk['chunk_id'] = f"{document['id']}_chunk_{i}"
            chunk['source'] = document.get('source', 'unknown')
            chunk['timestamp'] = document.get('timestamp')
        
        return chunks
```

### **7.2 Quality Control va Monitoring**

```python
def chunking_quality_monitor(chunks):
    """Chunking sifatini real-time monitoring"""
    
    warnings = []
    
    # Size checks
    chunk_sizes = [len(chunk['text'].split()) for chunk in chunks]
    avg_size = np.mean(chunk_sizes)
    
    if avg_size < 50:
        warnings.append("Chunk sizes too small - information loss possible")
    elif avg_size > 1000:
        warnings.append("Chunk sizes too large - retrieval precision may decrease")
    
    # Empty chunks
    empty_chunks = sum(1 for chunk in chunks if len(chunk['text'].strip()) == 0)
    if empty_chunks > 0:
        warnings.append(f"Found {empty_chunks} empty chunks")
    
    # Duplicate detection
    texts = [chunk['text'] for chunk in chunks]
    if len(texts) != len(set(texts)):
        warnings.append("Duplicate chunks detected")
    
    return warnings
```

## Dars xulosasi

Chunking - RAG tizimining eng muhim bosqichlaridan biri. To'g'ri chunking strategiyasi retrieval sifatini sezilarli darajada oshiradi.

**Asosiy xulosalar:**
- Document type ga mos preprocessing zarur
- Chunking strategiyasi task ga bog'liq tanlanishi kerak
- Semantic coherence muhim ahamiyatga ega
- Production da scalability va quality control zarur

## Keyingi darsga tayyorgarlik

Keyingi darsda retrieval algoritmlari va semantic search texnikalarini o'rganamiz.

## Amaliy vazifa

1. PDF dan table extraction qilish uchun qaysi tool eng yaxshi?
2. O'zbek tili uchun qanday chunking muammolari bo'lishi mumkin?
3. 1000 sahifali kitob uchun qaysi chunking strategiyasini tanlaysiz?
4. Chunk size ni qanday optimize qilish mumkin?