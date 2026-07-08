import re


def solve_ner(text: str) -> str:
    candidates = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text)
    if not candidates:
        return "No named entities found."
    return "Named entities: " + ", ".join(dict.fromkeys(candidates))
