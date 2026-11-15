from groundwork.evaluation import (
    EvalQuestion,
    Run,
    admits_not_found,
    best_interval_midpoint,
    gate_errors,
    is_relevant,
    retrieval_metrics,
)
from groundwork.retriever import Hit
from tests.conftest import make_chunk


def q(id_, type_="answerable", evidence=("QLoRA",), sources=(), style="short"):
    return EvalQuestion(id_, type_, "?", style, tuple(evidence), tuple(sources))


def test_evidence_matches_whole_words_only():
    question = q("dapt", evidence=("DAPT",))
    assert is_relevant(make_chunk("We call this DAPT, domain-adaptive pretraining."), question)
    assert not is_relevant(make_chunk("Adapter layers adapt the model."), question)


def test_evidence_matches_across_line_breaks():
    question = q("dpo", evidence=("Direct Preference Optimization",))
    assert is_relevant(make_chunk("simpler DPO (Direct\n  Preference Optimization, 2023)"), question)


def test_evidence_can_be_restricted_to_sources():
    question = q("kd", evidence=("student",), sources=("kd-notes",))
    assert is_relevant(make_chunk("the student model", "kd-notes"), question)
    assert not is_relevant(make_chunk("the student model", "tutoring"), question)


def test_retrieval_metrics_hit_rates_and_mrr():
    relevant, other = make_chunk("QLoRA uses 4-bit weights"), make_chunk("unrelated text")
    runs = [
        Run(q("first"), [Hit(relevant, 0.8), Hit(other, 0.5)]),
        Run(q("second"), [Hit(other, 0.8), Hit(relevant, 0.7)]),
        Run(q("missed"), [Hit(other, 0.8)]),
    ]
    m = retrieval_metrics(runs)["all"]
    assert m["hit@1"] == 1 / 3 and m["hit@k"] == 2 / 3
    assert m["mrr"] == (1 + 0.5 + 0) / 3


def test_gate_errors_count_false_refusals_and_false_accepts():
    chunk = make_chunk("text")
    runs = [
        Run(q("ok"), [Hit(chunk, 0.7)]),
        Run(q("short", style="short"), [Hit(chunk, 0.52, keyword=1.0)]),
        Run(q("offtopic", "out_of_scope"), [Hit(chunk, 0.55, keyword=0.2)]),
    ]
    assert gate_errors(runs, 0.6, 9.0) == (["short"], [])  # keyword rescue off
    assert gate_errors(runs, 0.6, 0.8) == ([], [])  # keyword rescue on
    assert gate_errors(runs, 0.5, 9.0) == ([], ["offtopic"])  # threshold too low


def test_best_interval_midpoint_picks_the_middle_of_the_widest_best_run():
    values = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
    errors = [3, 1, 0, 0, 0, 2]
    assert best_interval_midpoint(values, errors) == (0.65, 0)


def test_admits_not_found_recognizes_honest_wording():
    assert admits_not_found("The provided sources do not specify the exact GPU models.")
    assert admits_not_found("I don't have enough information in my documents to answer that.")
    assert not admits_not_found("LoRA trains two low-rank matrices [1].")


def test_joint_calibration_puts_both_thresholds_inside_the_error_free_region():
    from groundwork.evaluation import calibrate_gate, point_errors

    points = [
        (0.75, 0.20, "answerable"),  # passes on similarity
        (0.52, 0.95, "answerable"),  # short question: passes only on keywords
        (0.55, 0.40, "out_of_scope"),
        (0.60, 0.50, "out_of_scope"),
    ]
    tau, beta, errors = calibrate_gate(points)
    assert errors == 0 and point_errors(points, tau, beta) == (0, 0)
    assert 0.60 < tau <= 0.75 and 0.50 < beta <= 0.95


def test_split_half_cv_reports_heldout_rates():
    from groundwork.evaluation import split_half_cv

    points = [(0.7 + i / 100, 0.2, "answerable") for i in range(10)]
    points += [(0.5 + i / 100, 0.3, "out_of_scope") for i in range(10)]
    cv = split_half_cv(points, splits=50)
    assert cv["splits"] == 50
    # The two groups are separated, so every split calibrates cleanly.
    assert cv["heldout_false_refusal_rate"] == 0 and cv["heldout_false_accept_rate"] == 0


def test_random_retriever_baseline():
    from groundwork.evaluation import random_hit_at_k

    chunks = [make_chunk("QLoRA here")] + [make_chunk(f"filler {i}") for i in range(9)]
    # One relevant chunk out of 10, five random picks: a 50% chance of hitting it.
    assert abs(random_hit_at_k(chunks, [q("qlora")], k=5) - 0.5) < 1e-9
