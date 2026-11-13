import pytest
from fastapi.testclient import TestClient

from groundwork.api import create_app
from groundwork.ingest.pipeline import ingest
from groundwork.llm import LLMError
from groundwork.rag import RagPipeline
from groundwork.retriever import Retriever
from tests.conftest import FakeLLM, make_chunk

CHUNKS = [
    make_chunk("LoRA freezes the pretrained weights and trains two small low rank matrices", "lora", page=4, section=None),
    make_chunk("Distillation trains a small student model to imitate a large teacher model", "kd"),
]


@pytest.fixture
def client_with(embedder, store):
    ingest(CHUNKS, embedder, store)

    def make(llm) -> TestClient:
        pipeline = RagPipeline(Retriever(embedder, store), llm, top_k=2, min_similarity=0.3)
        return TestClient(create_app(pipeline))

    return make


def test_ask_returns_answer_sources_and_retrieval_details(client_with):
    with client_with(FakeLLM("LoRA trains small matrices [1].")) as client:
        res = client.post("/api/ask", json={"question": "How does LoRA train low rank matrices?"})
    assert res.status_code == 200
    data = res.json()
    assert data["refused"] is False and data["model"] == "fake/llm"
    assert data["sources"][0]["number"] == 1
    assert data["sources"][0]["chunk"]["location"] == "p. 4"
    assert [r["rank"] for r in data["retrieved"]] == [1, 2]
    assert data["retrieved"][0]["used"] is True
    assert data["retrieved"][0]["keyword"] is None  # this test pipeline is dense-only
    assert "retrieval_ms" in data["timings"]


def test_off_topic_question_is_refused_without_the_model(client_with):
    llm = FakeLLM("unused")
    with client_with(llm) as client:
        data = client.post("/api/ask", json={"question": "Capital of Australia?"}).json()
    assert data["refused"] is True and data["refusal_reason"] == "low_similarity"
    assert data["model"] is None and llm.calls == []


def test_empty_question_is_rejected(client_with):
    with client_with(FakeLLM("x")) as client:
        assert client.post("/api/ask", json={"question": ""}).status_code == 422


def test_model_failure_is_reported_as_service_unavailable(client_with):
    class BrokenLLM(FakeLLM):
        def generate(self, system, prompt):
            raise LLMError("Ollama is not reachable")

    with client_with(BrokenLLM("x")) as client:
        res = client.post("/api/ask", json={"question": "How does LoRA train low rank matrices?"})
    assert res.status_code == 503
    assert "not reachable" in res.json()["detail"]


def test_health_and_documents(client_with):
    with client_with(FakeLLM("x")) as client:
        health = client.get("/api/health").json()
        docs = client.get("/api/documents").json()
    assert health["chunks"] == 2 and health["llm_ok"] is True
    assert {d["source_id"] for d in docs} == {"lora", "kd"}


def test_web_page_is_served(client_with):
    with client_with(FakeLLM("x")) as client:
        res = client.get("/")
    assert res.status_code == 200 and "Retrieval inspector" in res.text
