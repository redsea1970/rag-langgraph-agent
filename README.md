# RAG LangGraph Agent

Агентный RAG: система, которая сама решает, что делать — искать в документах, переформулировать запрос или генерировать ответ.

## Архитектура (циклический граф)
- agent — принимает решение (retrieve / improve / generate).
- retrieve — ищет в документах.
- improve — переформулирует запрос, если поиск не дал результата.
- generate — генерирует финальный ответ.

## Что внутри
- data/docs/ — документы (Python, Git).
- data/questions.json — 5 тестовых вопросов.
- src/agent.py — реализация графа (демо-режим без LangGraph API).
- results/answers.txt — примеры диалогов.

## Как запустить
python3 src/agent.py

## Режимы
- Demo — граф на чистом Python, без API (по умолчанию).
- Live — настоящий LangGraph + OpenAI API.

## Автор
redsea1970
