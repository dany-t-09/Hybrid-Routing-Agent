def solve_summary(text: str) -> str:
    sentences = [part.strip() for part in text.replace("\n", " ").split(".") if part.strip()]
    if not sentences:
        return "Nothing to summarize."
    return sentences[0] + "."
