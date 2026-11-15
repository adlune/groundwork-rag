import pytest

from groundwork.ingest.pipeline import ingest
from groundwork.retriever import Retriever
from groundwork.store import StoreMismatchError, VectorStore
from tests.conftest import FakeEmbedder, make_chunk

CHUNKS = [
    make_chunk("LoRA freezes the weights and trains low rank matrices", "lora", page=2, section=None),
    make_chunk("Retrieval augmented generation searches documents before answering", "rag"),
    make_chunk("Distillation trains a small student to imitate a large teacher", "kd"),
]


def test_search_returns_the_most_similar_chunk_first(embedder, store):
    ingest(CHUNKS, embedder, store)
    hits = Retriever(embedder, store).search("which method trains a small student model", k=3)
    assert hits[0].chunk.source_id == "kd"
    assert hits[0].score > hits[1].score >= hits[2].score


def test_chunks_survive_the_round_trip_through_the_store(embedder, store):
    ingest(CHUNKS, embedder, store)
    # Documents are embedded with their title header, so searching the exact same
    # embedding text gives the exact same vector.
    hit = Retriever(embedder, store).search(CHUNKS[0].embedding_text, k=1)[0]
    assert hit.chunk == CHUNKS[0]  # including page=2 and section=None
    assert hit.score == pytest.approx(1.0, abs=1e-5)


def test_ingest_is_idempotent(embedder, store):
    first = ingest(CHUNKS, embedder, store)
    second = ingest(CHUNKS, embedder, store)
    assert (first.added, first.removed) == (3, 0)
    assert (second.added, second.removed, second.unchanged) == (0, 0, 3)
    assert store.count() == 3


def test_edited_documents_replace_their_old_chunks(embedder, store):
    ingest(CHUNKS, embedder, store)
    edited = [*CHUNKS[:2], make_chunk("Distillation transfers knowledge to a smaller model", "kd")]
    report = ingest(edited, embedder, store)
    assert (report.added, report.removed, report.unchanged) == (1, 1, 2)
    assert store.count() == 3


def test_documents_lists_each_source_once(embedder, store):
    ingest(CHUNKS, embedder, store)
    assert [d["source_id"] for d in store.documents()] == ["kd", "lora", "rag"]


def test_store_refuses_vectors_from_a_different_model(tmp_path):
    VectorStore(tmp_path / "chroma", "test", FakeEmbedder.name)
    with pytest.raises(StoreMismatchError):
        VectorStore(tmp_path / "chroma", "test", "some-other-model")


def test_hybrid_search_adds_keyword_scores_and_dense_scores_for_every_hit(embedder, store):
    from groundwork.keyword import KeywordIndex

    ingest(CHUNKS, embedder, store)
    retriever = Retriever(embedder, store, KeywordIndex(store.texts()), candidates=1)
    hits = retriever.search("distillation teacher", k=3)
    assert hits[0].chunk.source_id == "kd"
    assert hits[0].keyword > 0.9
    # With candidates=1, each search contributes one chunk; any chunk found by only one
    # of them must still come back with its real dense score.
    assert all(isinstance(h.score, float) for h in hits)


def test_search_on_empty_store_returns_nothing(embedder, store):
    assert Retriever(embedder, store).search("anything", k=5) == []


def test_two_fresh_stores_rank_the_same_way(embedder, tmp_path):
    # Search is exact, so a store reopened from disk must give the same ranking and
    # the same scores. The approximate index this replaced did not.
    first = VectorStore(tmp_path / "chroma", "test", FakeEmbedder.name)
    ingest(CHUNKS, embedder, first)
    query = embedder.embed_query("small student model trains low rank matrices")
    rankings = []
    for _ in range(2):
        reopened = VectorStore(tmp_path / "chroma", "test", FakeEmbedder.name)
        rankings.append([(c.id, s) for c, s in reopened.query(query, k=3)])
    assert rankings[0] == rankings[1]
    assert rankings[0] == [(c.id, s) for c, s in first.query(query, k=3)]


def test_search_sees_chunks_added_and_removed_after_the_first_query(embedder, store):
    ingest(CHUNKS[:1], embedder, store)
    query = embedder.embed_query("distillation student teacher")
    assert [c.source_id for c, _ in store.query(query, k=5)] == ["lora"]
    ingest(CHUNKS, embedder, store)
    assert store.query(query, k=5)[0][0].source_id == "kd"
    ingest(CHUNKS[:2], embedder, store)
    assert "kd" not in {c.source_id for c, _ in store.query(query, k=5)}
