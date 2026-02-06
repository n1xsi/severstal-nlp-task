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

    def _mock_llm_generation(self, prompt: str) -> str:
        """
        Имитация вызова LLM (GPT-4/GigaChat/Llama 3), так как это было разрешено в условиях задачи.
        В реальном проекте здесь был бы HTTP запрос к API.
        """
        # ПРИМЕР ЗАПРОСА API К ИСПОЛЬЗОВАНИЮ OPENAI:
        # response = openai.ChatCompletion.create(
        #     model="gpt-4",
        #     messages=[{"role": "user", "content": prompt}]
        # )
        # return response.choices[0].message.content

        return (
            "ОТВЕТ LLM (MOCK): На основе найденных фрагментов, документ содержит "
            "информацию, релевантную вашему запросу. Пожалуйста, подключите реальную "
            "модель (например, GPT-4 или Llama 3) в методе _mock_llm_response для "
            "генерации связного текста."
        )

    def answer_query(self, query: str, top_k: int = 3) -> str:
        """
        Полный цикл RAG (Retrieval-Augmented Generation): Вопрос -> Вектор -> Поиск -> Промпт -> Ответ.
        """
        if not self.chunks or self.embeddings is None:
            return "База знаний пуста. Сначала выполните index_documents."

        # Векторизация запроса
        query_vec = self.encoder.encode([query])

        # Подсчет косинусного сходства (Dot product для нормализованных векторов)
        # Форма: (1, N_chunks)
        scores = cosine_similarity(query_vec, self.embeddings)[0]

        # Поиск индексов топ-K лучших совпадений
        # argsort сортирует по возрастанию, нужно взять последние K и перевернуть
        top_k_indices = scores.argsort()[-top_k:][::-1]

        # Извлечение текста из найденных фрагментов
        retrieved_chunks = [self.chunks[i] for i in top_k_indices]

        # Формирование промпта для LLM
        context_str = "\n\n----------\n\n".join(retrieved_chunks)
        prompt = (
            f"Используй только следующие фрагменты документов для ответа:\n"
            f"{context_str}\n\n"
            f"Вопрос: {query}\n"
            f"Ответ:"
        )

        # Отладка: вывод найденного контекста
        # print(f"\n[DEBUG] Найден контекст для вопроса '{query}':")
        # print(f"Найденные фрагменты:\n{context_str}\n")

        return self._mock_llm_generation(prompt)
