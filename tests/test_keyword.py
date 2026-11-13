from groundwork.keyword import KeywordIndex, tokenize

DOCS = {
    "lora": "LoRA freezes the weights and trains low rank matrices.",
    "qlora": "QLoRA loads the base model in 4-bit precision and trains LoRA adapters.",
    "kd": "Distillation trains a small student model from a large teacher model.",
    "rag": "RAG retrieves documents for the model before it answers.",
}


def test_tokenize_drops_stopwords_and_punctuation():
    assert tokenize("What is QLoRA?") == ["qlora"]
    assert tokenize("How does the 4-bit model work") == ["bit", "model", "work"]


def test_rare_exact_term_ranks_first_with_full_strength():
    results = KeywordIndex(DOCS).search("What is QLoRA?", k=3)
    assert results[0][0] == "qlora"
    # Below 1.0 only because this chunk is longer than average (length normalization).
    assert results[0][1] > 0.85


def test_words_missing_from_the_corpus_lower_the_strength():
    index = KeywordIndex(DOCS)
    full = index.search("teacher student", k=1)[0][1]
    partial = index.search("teacher student football worldcup", k=1)[0][1]
    assert partial < full


def test_common_words_count_less_than_rare_words():
    index = KeywordIndex(DOCS)
    assert index.idf("model") < index.idf("qlora")


def test_no_match_returns_nothing():
    assert KeywordIndex(DOCS).search("capital of Australia", k=3) == []
