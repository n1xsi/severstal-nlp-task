# severstal-nlp-task

Реализация задачи 8 (DocumentAssistant) с использованием подхода RAG для IT Hub «Северстали».

## Стек технологий
* Python 3.10+
* Sentence-Transformers (Embeddings)
* NumPy/Scikit-Learn (vector search)
* PyPDF & Python-Docx (parsing)

## Запуск
```bash
pip install -r requirements.txt
python main.py
```

В первый раз запуск программы будет скачиваться модель (около 80 МБ). В конце работы в корне проекта будет создан файл `assistant_results.json`.

## Особенности:
* Извлечение текста из PDF/DOCX/TXT.
* Семантический поиск с использованием косинусоидального сходства (`cosine_similarity`).
* Настраиваемая стратегия разбиения на фрагменты.
