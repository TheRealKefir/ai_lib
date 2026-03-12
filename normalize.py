import re
from typing import Optional

def normalize_query(query: str) -> str:
    query = re.sub(r'[^\w\s.,!?;:()\-]', '', query, flags=re.UNICODE)
    query = re.sub(r'\s+', ' ', query).strip()
    return query

def build_prompt(
    user_query: str,
    book_context: Optional[str] = None,
    system_instruction: Optional[str] = None
) -> str:
    cleaned_query = normalize_query(user_query)

    if system_instruction is None:
        system_instruction = (
            "Ты — ассистент, анализирующий загруженные книги. "
            "Отвечай на вопросы, используя только информацию из предоставленного контекста. "
            "Если ответ невозможно найти в контексте, честно скажи, что не знаешь."
        )

    if book_context:
        prompt = (
            f"{system_instruction}\n\n"
            f"Контекст из книг:\n{book_context}\n\n"
            f"Вопрос: {cleaned_query}"
        )
    else:
        prompt = (
            f"{system_instruction}\n\n"
            f"Вопрос: {cleaned_query}"
        )

    return prompt

if __name__ == "__main__":
    user_input = "  Какие /   основные []идеи   в# книге   «1984»? "
    book_context = "В романе «1984» Джорджа Оруэлла описывается тоталитарное общество, где правящая партия во главе с Большим Братом контролирует каждый аспект жизни людей. Основные идеи: подавление индивидуальности, манипуляция историей, всеобщая слежка."
    pr = build_prompt(user_input, book_context=book_context)
    print(pr)