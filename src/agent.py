import json
import re
import math
import os
from collections import Counter

# === 1. Загрузка документов и TF-IDF ===
def load_docs(docs_dir="data/docs"):
    chunks = []
    for filename in os.listdir(docs_dir):
        if not filename.endswith(".md"):
            continue
        with open(os.path.join(docs_dir, filename), "r", encoding="utf-8") as f:
            text = f.read()
        for i, p in enumerate([p.strip() for p in text.split("\n\n") if p.strip()]):
            chunks.append({"id": f"{filename}::{i}", "source": filename, "text": p})
    return chunks

def tokenize(text):
    return re.findall(r"\w+", text.lower())

def build_index(chunks):
    df = Counter()
    for c in chunks:
        for t in set(tokenize(c["text"])):
            df[t] += 1
    return df

def score(query_tokens, chunk, df, total):
    tf = Counter(tokenize(chunk["text"]))
    return sum(tf[qt] * (math.log((total + 1) / (df.get(qt, 0) + 1)) + 1) for qt in query_tokens if qt in tf)

def retrieve(query, chunks, df, top_k=2):
    q_tokens = tokenize(query)
    scored = [(score(q_tokens, c, df, len(chunks)), c) for c in chunks]
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]

# === 2. Состояние агента ===
class AgentState:
    def __init__(self, question):
        self.question = question
        self.current_query = question
        self.retrieved = []
        self.answer = None
        self.attempts = 0
        self.log = []

    def log_event(self, node, message):
        self.log.append(f"[{node}] {message}")

# === 3. Узлы графа ===
def node_agent(state, chunks, df):
    """Решает, что делать: retrieve → generate."""
    state.log_event("agent", f"Анализ вопроса: {state.current_query}")
    if state.attempts == 0:
        state.log_event("agent", "Решение: retrieve")
        return "retrieve"
    else:
        state.log_event("agent", "Данные уже есть, решение: generate")
        return "generate"

def node_retrieve(state, chunks, df):
    """Ищет документы."""
    state.log_event("retrieve", f"Поиск по запросу: {state.current_query}")
    state.retrieved = retrieve(state.current_query, chunks, df)
    top_score = state.retrieved[0][0] if state.retrieved else 0
    state.log_event("retrieve", f"Найдено {len(state.retrieved)} чанков, top_score={top_score:.2f}")
    if top_score < 2.0:
        state.log_event("retrieve", "Низкая релевантность → improve")
        return "improve"
    state.log_event("retrieve", "Достаточно релевантно → generate")
    return "generate"

def node_improve(state, chunks, df):
    """Переформулирует запрос."""
    state.attempts += 1
    state.log_event("improve", f"Переформулировка (попытка {state.attempts})")
    state.current_query = state.question + " (уточнение: " + " ".join(tokenize(state.question)[:2]) + ")"
    return "retrieve"

def node_generate(state, chunks, df):
    """Генерирует ответ."""
    state.log_event("generate", "Формирование ответа")
    if not state.retrieved or state.retrieved[0][0] == 0:
        state.answer = "Не найдено релевантных документов."
    else:
        top = state.retrieved[0][1]
        state.answer = f"На основе {top['source']}: {top['text'][:150]}..."
    return "end"

# === 4. Запуск графа ===
def run_agent(question, chunks, df):
    state = AgentState(question)
    current_node = "agent"
    max_steps = 10
    step = 0

    while current_node != "end" and step < max_steps:
        step += 1
        if current_node == "agent":
            current_node = node_agent(state, chunks, df)
        elif current_node == "retrieve":
            current_node = node_retrieve(state, chunks, df)
        elif current_node == "improve":
            current_node = node_improve(state, chunks, df)
        elif current_node == "generate":
            current_node = node_generate(state, chunks, df)
        else:
            break

    return state

# === 5. Основной запуск ===
if __name__ == "__main__":
    chunks = load_docs()
    print(f"Загружено чанков: {len(chunks)}")
    df = build_index(chunks)

    with open("data/questions.json", "r", encoding="utf-8") as f:
        questions = json.load(f)

    os.makedirs("results", exist_ok=True)
    with open("results/answers.txt", "w", encoding="utf-8") as f:
        for q in questions:
            print("\n" + "=" * 60)
            print(f"Вопрос: {q['question']}")
            print("=" * 60)
            f.write("=" * 60 + "\n")
            f.write(f"Вопрос: {q['question']}\n")
            f.write("=" * 60 + "\n")

            state = run_agent(q["question"], chunks, df)
            for log in state.log:
                print(log)
                f.write(log + "\n")
            print(f"\n→ ОТВЕТ: {state.answer}")
            f.write(f"\n→ ОТВЕТ: {state.answer}\n\n")

    print("\nРезультаты сохранены в results/answers.txt")
