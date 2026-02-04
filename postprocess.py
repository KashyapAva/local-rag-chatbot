import re

def clean_answer(text: str) -> str:
    # Remove phrases like:
    # "This is supported by the information in [file#chunk], which states,"
    text = re.sub(
        r"(this is supported by|based on|according to)[^.,]*?\[[^\]]+?\][^.,]*?,",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # Remove any remaining [file#chunk]
    text = re.sub(r"\[[^\]]+?#chunk\d+\]", "", text)

    return text.strip()

