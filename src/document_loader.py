from pathlib import Path
from pypdf import PdfReader
from docx import Document


def load_text_from_file(file_path: str) -> str:
    """
    Извлекает текст из файлов формата *.pdf, *.docx, *.txt.
    Возвращает очищенную строку.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {file_path}")

    extension = path.suffix.lower()

    text = ""
    try:
        match extension:
            case '.pdf':
                # Сбор текста из всех страниц PDF, игнорируя страницы без текста
                reader = PdfReader(path)
                text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

            case '.docx':
                # Сбор текста из всех параграфов DOCX, игнорируя параграфы без текста
                doc = Document(path)
                text = "\n".join([para.text for para in doc.paragraphs if para.text])

            case '.txt':
                # Простое чтение текста из обычного текстового файла
                with open(path, 'r', encoding='utf-8') as f:
                    text = f.read()
            case _:
                print(f"Файл формата {extension} не поддерживается - файл пропущен.")
                return ""

    except Exception as e:
        print(f"Ошибка при чтении {file_path}: {e}")
        return ""

    # Базовая очистка от лишних пробелов
    return text.strip()
