from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from typing import List, Optional
import numpy as np


class DocumentAssistant:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Инициализация ассистента - загрузка модели эмбеддингов и подготовка данных.
        model_name: Название модели из HuggingFace (по умолчанию лёгкая и быстрая - all-MiniLM-L6-v2).
        """
        print(f"Загрузка модели {model_name}...")
        self.encoder = SentenceTransformer(model_name)
        self.chunks: List[str] = []
        self.embeddings: Optional[np.ndarray] = None

        # Параметры чанкинга
        self.chunk_size = 500  # Количество символов
        self.overlap = 50      # Перекрытие для сохранения контекста

    def _split_text_into_chunks(self, text: str) -> List[str]:
        """Разбивает текст на фрагменты с перекрытием."""
        if not text:
            return []

        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = min(start + self.chunk_size, text_len)
            chunk = text[start:end]

            # Удаление перекрытия из контекста
            if len(chunk) > self.overlap or not chunks:
                chunks.append(chunk)
            start += self.chunk_size - self.overlap

        return chunks

    def index_documents(self, documents: List[str]) -> None:
        """
        Функция разбивает документы на чанки и вычисляет векторные представления (embeddings).
        documents: Список строк документов.
        """
        self.chunks = []

        # Чанкинг
        print("Разбиение документов на чанки...")
        for doc in documents:
            doc_chunks = self._split_text_into_chunks(doc)
            self.chunks.extend(doc_chunks)

        if not self.chunks:
            print("Предупреждение: Нет данных для индексации.")
            return

        # Векторизация
        print(f"Вычисление эмбеддингов для {len(self.chunks)} фрагментов...")
        self.embeddings = self.encoder.encode(self.chunks, show_progress_bar=True)
        print("Индексация завершена.")
