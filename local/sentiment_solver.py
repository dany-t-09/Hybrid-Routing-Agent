POSITIVE_WORDS = {"good", "great", "excellent", "happy", "love", "awesome", "positive"}
NEGATIVE_WORDS = {"bad", "terrible", "sad", "hate", "awful", "negative", "poor"}


def solve_sentiment(text: str) -> str:
    words = {word.strip(".,!?;:").lower() for word in text.split()}
    positive_score = len(words & POSITIVE_WORDS)
    negative_score = len(words & NEGATIVE_WORDS)

    if positive_score > negative_score:
        return "Sentiment: positive"
    if negative_score > positive_score:
        return "Sentiment: negative"
    return "Sentiment: neutral"
