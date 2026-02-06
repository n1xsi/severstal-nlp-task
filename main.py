from src.document_loader import load_text_from_file
from src.assistant import DocumentAssistant

import json
import os


def main():
    print("----------Запуск Document Assistant...----------")

    # Конфигурация путей
    files_to_process = [
        "data/A9RD3D4.pdf",
        "data/Polzovatelskoe_soglashenie.pdf",
        "data/University Success.docx"
    ]

    # Загрузка данных
    raw_documents = []
    print("---------- ЗАГРУЗКА ДОКУМЕНТОВ ----------")
    for file_path in files_to_process:
        if os.path.exists(file_path):
            text = load_text_from_file(file_path)
            if text:
                raw_documents.append(text)
                print(f"Файл {file_path} успешно загружен ({len(text)} символов).")
        else:
            print(f"Ошибка: Файл {file_path} не найден.")

    if not raw_documents:
        print("Ошибка: Нет документов для обработки.")
        return

    # Инициализация и индексация
    print("\n---------- ИНИЦИАЛИЗАЦИЯ AI ----------")
    assistant = DocumentAssistant()
    assistant.index_documents(raw_documents)

    # Тестирование (по вопросам)
    questions = [
        "О чем пользовательское соглашение?",
        "Какие критерии успеха университета?",
        "Что содержится в документе A9RD3D4?"
    ]

    # Ответы на вопросы
    results = {}
    print("\n---------- ГЕНЕРАЦИЯ ОТВЕТОВ ----------")
    for q in questions:
        answer = assistant.answer_query(q)
        results[q] = answer
        print(f"Question: {q}\nAnswer: {answer}")

    # Сохранение результатов
    output_file = "assistant_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"Результаты работы сохранены в {output_file}")


if __name__ == "__main__":
    main()
