"conversational": """Previous conversation:
{conversation_history}

Current context:
{context}

User: {query}

Please respond naturally, taking into account both the conversation history and the current context. If the current context provides new information that updates or contradicts previous information, acknowledge this appropriately.""",

            "fact_checking": """Context: {context}

Question: {query}

Instructions:
1. Carefully examine the provided context
2. Identify specific facts that support or contradict the question
3. Cite specific parts of the context in your answer
4. If there are any uncertainties or limitations in the information, mention them

Fact-based Answer:""",

            "creative_writing": """Context Information: {context}

Creative Task: {query}

Guidelines:
- Use the provided context as inspiration and factual foundation
- Be creative while staying true to the core information
- Maintain accuracy for any factual elements
- Clearly distinguish between factual content and creative additions

Creative Response:""",

            "technical_explanation": """Technical Context: {context}

Technical Question: {query}

Please provide a technical explanation that:
1. Uses precise terminology from the context
2. Explains concepts step-by-step
3. Includes relevant technical details
4. Provides examples where appropriate

Technical Answer:"""
        }
    
    def get_template(self, template_name: str) -> str:
        """Template olish"""
        return self.templates.get(template_name, self.templates["basic_qa"])
    
    def format_prompt(self, template_name: str, **kwargs) -> str:
        """Template formatlab prompt yaratish"""
        template = self.get_template(template_name)
        return template.format(**kwargs)
    
    def add_custom_template(self, name: str, template: str):
        """Yangi template qo'shish"""
        self.templates[name] = template

# Chain-of-Thought prompting
class ChainOfThoughtPrompter:
    def __init__(self):
        self.cot_template = """Context: {context}

Question: {query}

Let's think through this step by step:

1. First, let me identify the key information from the context that relates to this question:
[Relevant information identification]

2. Now, let me analyze what the question is specifically asking:
[Question analysis]

3. Based on the context, here's my reasoning:
[Step-by-step reasoning]

4. Therefore, my answer is:
[Final answer]

Please follow this reasoning structure in your response."""
    
    def create_cot_prompt(self, context: str, query: str) -> str:
        return self.cot_template.format(context=context, query=query)

# Few-shot prompting
class FewShotPrompter:
    def __init__(self):
        self.examples = []
    
    def add_example(self, context: str, query: str, expected_answer: str):
        """Few-shot example qo'shish"""
        self.examples.append({
            "context": context,
            "query": query,
            "answer": expected_answer
        })
    
    def create_few_shot_prompt(self, context: str, query: str) -> str:
        """Few-shot prompt yaratish"""
        prompt = "Here are some examples of how to answer questions based on context:\n\n"
        
        for i, example in enumerate(self.examples[-3:], 1):  # Oxirgi 3ta example
            prompt += f"Example {i}:\n"
            prompt += f"Context: {example['context']}\n"
            prompt += f"Question: {example['query']}\n"
            prompt += f"Answer: {example['answer']}\n\n"
        
        prompt += "Now, please answer the following question using the same approach:\n\n"
        prompt += f"Context: {context}\n"
        prompt += f"Question: {query}\n"
        prompt += f"Answer:"
        
        return prompt
```

---

## 3. Context Management

### 3.1 Context Filtering va Ranking
```python
import re
from typing import List, Tuple

class ContextManager:
    def __init__(self, max_context_length=4000):
        self.max_context_length = max_context_length
    
    def filter_relevant_sentences(self, context: str, query: str, 
                                 threshold: float = 0.3) -> str:
        """Query bilan bog'liq sentencelarni filterlash"""
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity
        
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Sentencega bo'lish
        sentences = re.split(r'[.!?]+', context)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return context
        
        # Embeddings
        query_embedding = model.encode([query])
        sentence_embeddings = model.encode(sentences)
        
        # Similarity hisoblash
        similarities = cosine_similarity(query_embedding, sentence_embeddings)[0]
        
        # Relevant sentencelarni tanlab olish
        relevant_sentences = []
        for sentence, similarity in zip(sentences, similarities):
            if similarity >= threshold:
                relevant_sentences.append((sentence, similarity))
        
        # Similarity bo'yicha saralash
        relevant_sentences.sort(key=lambda x: x[1], reverse=True)
        
        # Birlashtirib qaytarish
        if relevant_sentences:
            return '. '.join([s[0] for s in relevant_sentences])
        else:
            # Agar hech narsa topilmasa, eng yaqin sentencelarni qaytarish
            top_indices = similarities.argsort()[-3:][::-1]
            return '. '.join([sentences[i] for i in top_indices])
    
    def truncate_context(self, context: str, query: str) -> str:
        """Context uzunligini cheklash"""
        if len(context) <= self.max_context_length:
            return context
        
        # Relevant qismlarni birinchi navbatda saqlash
        filtered_context = self.filter_relevant_sentences(context, query)
        
        if len(filtered_context) <= self.max_context_length:
            return filtered_context
        
        # Agar hali ham uzun bo'lsa, oddiy truncation
        return filtered_context[:self.max_context_length] + "..."
    
    def merge_contexts(self, contexts: List[str], query: str, 
                      max_sources: int = 5) -> str:
        """Bir nechta kontekstni birlashtirish"""
        if not contexts:
            return ""
        
        # Har bir kontekst uchun relevance score hisoblash
        scored_contexts = []
        for i, context in enumerate(contexts[:max_sources]):
            # Simple scoring based on query keywords
            query_words = set(query.lower().split())
            context_words = set(context.lower().split())
            overlap = len(query_words.intersection(context_words))
            score = overlap / len(query_words) if query_words else 0
            
            scored_contexts.append((context, score, i))
        
        # Score bo'yicha saralash
        scored_contexts.sort(key=lambda x: x[1], reverse=True)
        
        # Eng yaxshi kontekstlarni birlashtirish
        merged = ""
        for context, score, source_idx in scored_contexts:
            addition = f"\n\nSource {source_idx + 1}:\n{context}"
            if len(merged + addition) <= self.max_context_length:
                merged += addition
            else:
                # Qolgan joyga sig'adigan qismini qo'shish
                remaining_space = self.max_context_length - len(merged)
                if remaining_space > 100:  # Kamida 100 char qolgan bo'lsa
                    merged += addition[:remaining_space] + "..."
                break
        
        return merged.strip()

# Hierarchical context management
class HierarchicalContextManager:
    def __init__(self):
        self.context_layers = {
            "immediate": [],      # Bevosita relevant
            "supporting": [],     # Qo'shimcha ma'lumot
            "background": []      # Umumiy kontekst
        }
    
    def categorize_context(self, contexts: List[str], query: str) -> Dict:
        """Kontekstlarni kategoriyalarga bo'lish"""
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity
        
        model = SentenceTransformer('all-MiniLM-L6-v2')
        query_embedding = model.encode([query])
        
        categorized = {
            "immediate": [],
            "supporting": [],
            "background": []
        }
        
        for context in contexts:
            context_embedding = model.encode([context])
            similarity = cosine_similarity(query_embedding, context_embedding)[0][0]
            
            if similarity > 0.7:
                categorized["immediate"].append(context)
            elif similarity > 0.4:
                categorized["supporting"].append(context)
            else:
                categorized["background"].append(context)
        
        return categorized
    
    def build_layered_context(self, categorized_contexts: Dict, 
                            max_length: int = 4000) -> str:
        """Qatlamli kontekst yaratish"""
        result = ""
        
        # Immediate context (eng muh            "conversational": """Previous conversation:
{conversation_history}

Current context:
{context}

User: {query}

Please respond naturally, taking into account both the conversation history and the current context. If the current context provides new information that updates or contradicts previous information, acknowledge this appropriately.            "conversational": """Previous conversation:
{conversation_history}

Current context:
{context}

User: {query}

Please respond naturally, taking into account both the conversation history and the current context. If the current context provides new information that updates or contradicts previous information, acknowledge this appropriately.            "conversational": """Previous conversation:
{conversation_history}

Current context:
{context}

User: {query}

Please respond naturally, taking into account both the conversation history and the current context. If the current context provides new information that updates or contradicts previous information, acknowledge this appropriately.            "conversational": """Previous conversation:
{conversation_history}

Current context:
{context}

User: {query}

Please respond naturally, taking into account both the conversation history and the current context. If the current context provides new information that updates or contradicts previous information, acknowledge this appropriately.            "conversational": """Previous conversation:
{conversation_history}

Current context:
{context}

User: {query}

Please respond naturally, taking into account both the conversation history and the current context. If the current context provides new information that updates or contradicts previous information, acknowledge this appropriately.# RAG Mastery Course - 6-dars
## Generation va Prompt Engineering

### 📚 Dars maqsadi
Bu darsda RAG sistemasining generation qismini, prompt engineering texnikalarini va context-aware javob yaratish usullarini chuqur o'rganasiz.

---

## 1. RAG Generation Fundamentals

### 1.1 Basic Generation Pipeline
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class RAGGenerator:
    def __init__(self, model_name="microsoft/DialoGPT-medium"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)
        
        # Tokenizer uchun pad token qo'shish
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def generate_response(self, context, query, max_length=512, temperature=0.7):
        """Context va query asosida javob generatsiya qilish"""
        
        # Prompt yaratish
        prompt = f"""Context: {context}
        
Question: {query}

Answer: """
        
        # Tokenization
        inputs = self.tokenizer.encode(prompt, return_tensors="pt", 
                                     max_length=max_length, truncation=True)
        inputs = inputs.to(self.device)
        
        # Generation
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=inputs.shape[1] + 150,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                top_k=50,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode qilish
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Faqat answer qismini ajratish
        answer_start = response.find("Answer:") + len("Answer:")
        answer = response[answer_start:].strip()
        
        return answer
```

### 1.2 OpenAI API bilan Generation
```python
import openai
from typing import List, Dict
import json

class OpenAIGenerator:
    def __init__(self, api_key, model="gpt-3.5-turbo"):
        openai.api_key = api_key
        self.model = model
        self.conversation_history = []
    
    def generate_response(self, context: str, query: str, 
                         system_prompt: str = None, 
                         temperature: float = 0.7) -> Dict:
        """OpenAI API orqali javob generatsiya qilish"""
        
        # Default system prompt
        if system_prompt is None:
            system_prompt = """You are a helpful AI assistant that answers questions based on the provided context. 
            Use only the information from the context to answer. If the context doesn't contain enough information 
            to answer the question, say so clearly."""
        
        # Messages tuzish
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        
        # Conversation history qo'shish (agar mavjud bo'lsa)
        messages.extend(self.conversation_history[-4:])  # Oxirgi 4ta message
        
        # Current context va query
        user_content = f"""Context:
{context}

Question: {query}"""
        
        messages.append({"role": "user", "content": user_content})
        
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=500,
                top_p=0.9,
                frequency_penalty=0.1,
                presence_penalty=0.1
            )
            
            answer = response.choices[0].message.content
            
            # Conversation history update qilish
            self.conversation_history.append({"role": "user", "content": user_content})
            self.conversation_history.append({"role": "assistant", "content": answer})
            
            return {
                "answer": answer,
                "usage": response.usage,
                "model": self.model,
                "temperature": temperature
            }
            
        except Exception as e:
            return {
                "answer": f"Xatolik yuz berdi: {str(e)}",
                "error": True
            }
    
    def clear_history(self):
        """Suhbat tarixini tozalash"""
        self.conversation_history = []
```

---

## 2. Advanced Prompt Engineering

### 2.1 Prompt Templates va Strategies
```python
class PromptTemplateManager:
    def __init__(self):
        self.templates = {
            "basic_qa": """Context: {context}

Question: {query}

Please provide a clear and accurate answer based only on the information provided in the context.

Answer:""",

            "analytical": """Given the following information:
{context}

Question: {query}

Instructions:
1. Analyze the provided information carefully
2. Extract relevant facts that relate to the question
3. Provide a well-reasoned answer
4. If the information is insufficient, clearly state what additional information would be needed

Analysis and Answer:""",

            "step_by_step": """Context Information:
{context}

User Question: {query}

Please follow these steps:
1. Identify the key information from the context that relates to the question
2. Break down the problem or question into components
3. Address each component systematically
4. Provide a clear, step-by-step answer

Step-by-step Response:""",

            "conversational": """Previous conversation:
{conversation_history}

Current context:
{context}

User: {query}