"""
Central configuration for the review clustering pipeline.
"""
import re
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
CSV_DIR = BASE_DIR.parent / "review_parser" / "results"
RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
EMBEDDINGS_CACHE_DIR = RESULTS_DIR / "embeddings_cache"
REPORTS_DIR = RESULTS_DIR / "reports"

for _d in (RESULTS_DIR, PLOTS_DIR, EMBEDDINGS_CACHE_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── Embedding models ────────────────────────────────────────
EMBEDDING_MODELS = {
    "bge-m3": {
        "name": "BAAI/bge-m3",
        "dim": 1024,
        "max_seq_length": 8192,
        "batch_size": 32,
        "prefix": None,
    },
    "e5-large": {
        "name": "intfloat/multilingual-e5-large",
        "dim": 1024,
        "max_seq_length": 512,
        "batch_size": 24,
        "prefix": "passage: ",
    },
    "rubert-tiny2": {
        "name": "cointegrated/rubert-tiny2",
        "dim": 312,
        "max_seq_length": 2048,
        "batch_size": 64,
        "prefix": None,
    },
}

DEFAULT_EMBEDDING_MODEL = "bge-m3"

# ── Topic naming models ──────────────────────────────────────
TOPIC_NAMING_MODELS = {
    "yandex-lite-sync": "yandex:yandexgpt-lite/latest",
    "yandex-pro-sync": "yandex:yandexgpt/latest",
    "local-qwen25-3b": "local:Qwen/Qwen2.5-3B-Instruct",
    "local-saiga-mistral-7b": "local:IlyaGusev/saiga_mistral_7b",
}
DEFAULT_TOPIC_NAMING_MODEL = "yandex-lite-sync"

# ── UMAP defaults ───────────────────────────────────────────
DEFAULT_UMAP_PARAMS = dict(
    n_neighbors=10,
    n_components=5,
    min_dist=0.0,
    metric="cosine",
    random_state=42,
    low_memory=True,
)

UMAP_VIS_PARAMS = dict(
    n_neighbors=15,
    n_components=2,
    min_dist=0.1,
    metric="cosine",
    random_state=42,
)

# ── HDBSCAN defaults ────────────────────────────────────────
DEFAULT_HDBSCAN_PARAMS = dict(
    min_cluster_size=5,
    min_samples=2,
    metric="euclidean",
    cluster_selection_method="eom",
    prediction_data=True,
)

# ── BERTopic / Vectorizer ───────────────────────────────────
TOP_N_WORDS = 10

_CYRILLIC_RE = re.compile("[\u0400-\u04ff]{2,}")

def cyrillic_tokenizer(text: str) -> list[str]:
    """Extract sequences of 2+ Cyrillic characters from already-lowercased text."""
    return _CYRILLIC_RE.findall(text)

VECTORIZER_PARAMS = dict(
    min_df=1,
    ngram_range=(1, 2),
    tokenizer=cyrillic_tokenizer,
)

# ── Preprocessing ────────────────────────────────────────────
MIN_TEXT_CHARS = 15
NEGATIVE_RATING_THRESHOLD = 3  

# ── Adaptive topic count ────────────────────────────────────
def get_target_topics(n_docs: int) -> int:
    if n_docs <= 50:
        return 3
    if n_docs <= 200:
        return 5
    if n_docs <= 600:
        return 8
    return 12

MAX_TOPICS = 15

# ── Stop-words ───────────────────────────────────────────────
RUSSIAN_STOP_WORDS: list[str] = [
    "и", "в", "во", "не", "что", "он", "на", "я", "с", "со",
    "как", "а", "то", "все", "она", "так", "его", "но", "да",
    "ты", "к", "у", "же", "вы", "за", "бы", "по", "только",
    "ее", "мне", "было", "вот", "от", "меня", "еще", "нет",
    "о", "из", "ему", "теперь", "когда", "даже", "ну", "вдруг",
    "ли", "если", "уже", "или", "ни", "быть", "был", "него",
    "до", "вас", "нибудь", "опять", "уж", "вам", "ведь", "там",
    "потом", "себя", "ничего", "ей", "может", "они", "тут",
    "где", "есть", "надо", "ней", "для", "мы", "тебя", "их",
    "чем", "была", "сам", "чтоб", "без", "будто", "чего", "раз",
    "тоже", "себе", "под", "будет", "ж", "тогда", "кто", "этот",
    "того", "потому", "этого", "какой", "совсем", "ним", "здесь",
    "этом", "один", "почти", "мой", "тем", "чтобы", "нее",
    "сейчас", "были", "куда", "зачем", "всех", "никогда",
    "можно", "при", "наконец", "два", "об", "другой", "хоть",
    "после", "над", "больше", "тот", "через", "эти", "нас",
    "про", "всего", "них", "какая", "много", "разве", "три",
    "эту", "моя", "впрочем", "хорошо", "свою", "этой", "перед",
    "иногда", "лучше", "чуть", "том", "нельзя", "такой", "им",
    "более", "всегда", "конечно", "всю", "между", "еще", "это",
    "этих", "этим", "этими", "та", "те", "тех", "тем", "теми",
    "той", "бы", "б", "ли", "ль", "мне", "мной", "мною",
    "нами", "ними", "нему", "нею", "нем", "ней",
    "достоинства", "недостатки", "комментарий",
]

MARKETPLACE_STOP_WORDS: list[str] = [
    "wb", "wildberries", "вайлдберриз", "вб", "валдберис",
    "озон", "ozon",
    "товар", "товара", "товаром", "товару", "товаре",
    "заказ", "заказа", "заказала", "заказал", "заказали",
    "доставка", "доставки", "доставили", "доставлен",
    "пришел", "пришла", "пришло", "пришли",
    "получила", "получил", "получили",
    "очень", "просто", "вообще", "конечно", "наверное",
    "купила", "купил", "купили", "брала", "брал", "взяла", "взял",
    "продавец", "продавца", "продавцу", "магазин", "магазина",
    "рекомендую", "советую", "рекомендовать",
    "спасибо", "пожалуйста", "благодарю",
    "фото", "фотография", "картинка", "фотографии",
    "отзыв", "отзывы", "отзыва", "отзывов",
    "звезда", "звезды", "звезд", "звёзды",
    "поставлю", "поставила", "ставлю",
]

ENGLISH_STOP_WORDS: list[str] = [
    "the", "is", "at", "which", "on", "a", "an", "and", "or", "not",
    "it", "to", "in", "he", "she", "for", "was", "are", "as", "with",
    "his", "they", "be", "this", "have", "from", "one", "had", "by",
    "but", "some", "what", "there", "we", "can", "out", "other", "were",
    "all", "do", "if", "will", "up", "about", "no", "so", "my", "of",
    "me", "its", "you", "your", "that", "love", "like", "just", "very",
    "good", "great", "nice", "super", "ok", "okay", "yes",
]

ALL_STOP_WORDS: list[str] = list(set(
    RUSSIAN_STOP_WORDS + MARKETPLACE_STOP_WORDS + ENGLISH_STOP_WORDS
))

# ── Text field columns in the CSV ────────────────────────────
TEXT_FIELDS = ["advantages", "disadvantages", "comment"]
TAGS_FIELD = "tags"
RATING_FIELD = "rating"
REVIEW_ID_FIELD = "review_id"
