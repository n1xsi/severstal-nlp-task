<h1 align="center">
  
  severstal-nlp-task

  [![Python](https://custom-icon-badges.demolab.com/badge/Python-white?style=for-the-badge&logo=pythonn)](#)
  [![NumPy](https://img.shields.io/badge/numpy-white?style=for-the-badge&logo=numpy&logoColor=013243)](#)
  [![scikit-learn](https://img.shields.io/badge/scikit--learn-white?style=for-the-badge&logo=scikit-learn&logoColor=f7931e)](#)
  
</h1>

Базовая реализация задачи 8 (DocumentAssistant) с использованием подхода RAG для IT Hub «Северстали».

Запуск LLM локально затруднена, поэтому использована mock-функция, имитирующая вызов модели (*в коде есть комментарий с примером того, как заменить модель на реальную*).

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

## Особенности
* Извлечение текста из PDF/DOCX/TXT.
* Семантический поиск с использованием косинусоидального сходства (`cosine_similarity`).
* Настраиваемая стратегия разбиения на фрагменты.
