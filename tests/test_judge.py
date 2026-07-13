"""Tests for the rubric-based LLM judge: rubric loading, the judge scorer,
and calibration against human labels.
"""

import json
from pathlib import Path

import pytest

from my_coding_agent.evals.judge import (
    CriterionVerdict,
    JudgeError,
    JudgeScorer,
    JudgeVerdict,
    LabelledCase,
    RubricError,
    calibrate,
    cohens_kappa,
    load_rubric,
    score_with_judge,
)
from my_coding_agent.evals.schema import EvalCase
from my_coding_agent.evals.scoring import RunResult, resolve_scorer

# --- fixtures ------------------------------------------------------------


def _write_rubric(tmp_path, data):
    path = tmp_path / "rubric.json"
    path.write_text(json.dumps(data))
    return path


def _well_formed_rubric_data():
    return {
        "name": "helpfulness",
        "scale": {"min": 1, "max": 5},
        "criteria": [
            {
                "name": "correctness",
                "description": "Is the answer factually correct?",
                "anchors": {"1": "completely wrong", "5": "fully correct"},
            },
            {
                "name": "tone",
                "description": "Is the tone appropriate?",
                "anchors": {"1": "rude or dismissive", "5": "clear and respectful"},
            },
        ],
    }


def _judge_json_response(criteria_scores, overall_score=4.0):
    body = {
        "criteria": {
            name: {"score": score, "rationale": f"rationale for {name}"}
            for name, score in criteria_scores.items()
        },
        "overall_score": overall_score,
        "overall_rationale": "overall rationale",
    }
    return json.dumps(body)


def _fake_llm(mocker, content, finish_reason="stop"):
    fake_llm = mocker.Mock()
    fake_llm.chat_completion.return_value = mocker.Mock(
        json=lambda: {
            "choices": [
                {
                    "message": {"content": content},
                    "finish_reason": finish_reason,
                }
            ]
        }
    )
    return fake_llm


# --- rubric loading --------------------------------------------------------


def test_load_rubric_well_formed(tmp_path):
    path = _write_rubric(tmp_path, _well_formed_rubric_data())
    rubric = load_rubric(path)
    assert rubric.name == "helpfulness"
    assert rubric.scale_min == 1
    assert rubric.scale_max == 5
    assert [c.name for c in rubric.criteria] == ["correctness", "tone"]
    assert rubric.criteria[0].anchors[1] == "completely wrong"


def test_load_rubric_missing_file_errors_clearly(tmp_path):
    with pytest.raises(RubricError, match="not found"):
        load_rubric(tmp_path / "missing.json")


def test_load_rubric_invalid_json_errors_clearly(tmp_path):
    path = tmp_path / "rubric.json"
    path.write_text("{not valid json")
    with pytest.raises(RubricError, match="not valid JSON"):
        load_rubric(path)


def test_load_rubric_missing_field_errors_clearly(tmp_path):
    data = _well_formed_rubric_data()
    del data["scale"]
    path = _write_rubric(tmp_path, data)
    with pytest.raises(RubricError, match="missing required field"):
        load_rubric(path)


def test_load_rubric_empty_criteria_errors(tmp_path):
    data = _well_formed_rubric_data()
    data["criteria"] = []
    path = _write_rubric(tmp_path, data)
    with pytest.raises(RubricError, match="at least one criterion"):
        load_rubric(path)


def test_load_rubric_inverted_scale_errors(tmp_path):
    data = _well_formed_rubric_data()
    data["scale"] = {"min": 5, "max": 1}
    path = _write_rubric(tmp_path, data)
    with pytest.raises(RubricError, match="scale.min"):
        load_rubric(path)


def test_load_rubric_out_of_scale_anchor_errors(tmp_path):
    data = _well_formed_rubric_data()
    data["criteria"][0]["anchors"]["9"] = "impossible"
    path = _write_rubric(tmp_path, data)
    with pytest.raises(RubricError, match="outside the rubric scale"):
        load_rubric(path)


# --- judge scorer ----------------------------------------------------------


def test_score_with_judge_produces_structured_verdict(mocker, tmp_path):
    rubric = load_rubric(_write_rubric(tmp_path, _well_formed_rubric_data()))
    content = _judge_json_response({"correctness": 5, "tone": 4})
    fake_llm = _fake_llm(mocker, content)

    verdict = score_with_judge(fake_llm, rubric, task="2+2?", output="4")

    assert isinstance(verdict, JudgeVerdict)
    assert verdict.overall_score == 4.0
    assert verdict.overall_rationale == "overall rationale"
    by_name = {c.name: c for c in verdict.criteria}
    assert by_name["correctness"] == CriterionVerdict(
        name="correctness", score=5, rationale="rationale for correctness"
    )
    assert by_name["tone"].score == 4

    _, kwargs = fake_llm.chat_completion.call_args
    assert kwargs["kind"] == "judge"


def test_score_with_judge_rationale_present_for_bias_inspection(mocker, tmp_path):
    """Both the per-criterion rationale and overall rationale must survive
    into the verdict so a verbose-but-wrong vs terse-correct pair can be
    told apart on inspection, not just by a single number.
    """
    rubric = load_rubric(_write_rubric(tmp_path, _well_formed_rubric_data()))
    verbose_llm = _fake_llm(
        mocker, _judge_json_response({"correctness": 2, "tone": 5}, overall_score=3.5)
    )
    terse_llm = _fake_llm(
        mocker, _judge_json_response({"correctness": 5, "tone": 3}, overall_score=4.0)
    )

    verbose_verdict = score_with_judge(
        verbose_llm, rubric, "q", "a long confident wrong answer"
    )
    terse_verdict = score_with_judge(terse_llm, rubric, "q", "short correct answer")

    for verdict in (verbose_verdict, terse_verdict):
        assert verdict.overall_rationale
        for c in verdict.criteria:
            assert c.rationale

    verbose_dict = verbose_verdict.to_dict()
    assert verbose_dict["criteria"][0]["rationale"]


def test_score_with_judge_malformed_json_raises(mocker, tmp_path):
    rubric = load_rubric(_write_rubric(tmp_path, _well_formed_rubric_data()))
    fake_llm = _fake_llm(mocker, "not json at all")
    with pytest.raises(JudgeError, match="no JSON object"):
        score_with_judge(fake_llm, rubric, "q", "a")


def test_score_with_judge_json_like_but_invalid_raises(mocker, tmp_path):
    rubric = load_rubric(_write_rubric(tmp_path, _well_formed_rubric_data()))
    fake_llm = _fake_llm(mocker, "here: {not: valid, json}")
    with pytest.raises(JudgeError, match="not valid JSON"):
        score_with_judge(fake_llm, rubric, "q", "a")


def test_score_with_judge_mismatched_criteria_raises(mocker, tmp_path):
    rubric = load_rubric(_write_rubric(tmp_path, _well_formed_rubric_data()))
    content = _judge_json_response({"correctness": 5})  # missing "tone"
    fake_llm = _fake_llm(mocker, content)
    with pytest.raises(JudgeError, match="do not match rubric criteria"):
        score_with_judge(fake_llm, rubric, "q", "a")


def test_score_with_judge_out_of_scale_score_raises(mocker, tmp_path):
    rubric = load_rubric(_write_rubric(tmp_path, _well_formed_rubric_data()))
    content = _judge_json_response({"correctness": 99, "tone": 4})
    fake_llm = _fake_llm(mocker, content)
    with pytest.raises(JudgeError, match="outside the rubric scale"):
        score_with_judge(fake_llm, rubric, "q", "a")


def test_score_with_judge_strips_think_tags_and_fences(mocker, tmp_path):
    rubric = load_rubric(_write_rubric(tmp_path, _well_formed_rubric_data()))
    raw = _judge_json_response({"correctness": 5, "tone": 5})
    wrapped = f"<think>reasoning...</think>\n```json\n{raw}\n```"
    fake_llm = _fake_llm(mocker, wrapped)

    verdict = score_with_judge(fake_llm, rubric, "q", "a")
    assert verdict.overall_score == 4.0


def test_score_with_judge_truncated_response_raises(mocker, tmp_path):
    rubric = load_rubric(_write_rubric(tmp_path, _well_formed_rubric_data()))
    raw = _judge_json_response({"correctness": 5, "tone": 5})
    fake_llm = _fake_llm(mocker, raw, finish_reason="length")
    with pytest.raises(JudgeError, match="truncated"):
        score_with_judge(fake_llm, rubric, "q", "a")


def test_score_with_judge_truncated_cot_example_not_mistaken_for_verdict(
    mocker, tmp_path
):
    # A truncated response whose unclosed <think> block contains an
    # illustrative example JSON snippet must never be parsed as the real
    # verdict — truncation is caught before any JSON extraction is attempted.
    rubric = load_rubric(_write_rubric(tmp_path, _well_formed_rubric_data()))
    example = _judge_json_response({"correctness": 1, "tone": 1}, overall_score=1.0)
    wrapped = (
        "<think>For example, a bad answer might score like this: "
        f"```json\n{example}\n``` Now let me actually grade the a"
    )
    fake_llm = _fake_llm(mocker, wrapped, finish_reason="length")
    with pytest.raises(JudgeError, match="truncated"):
        score_with_judge(fake_llm, rubric, "q", "a")


def test_extract_json_candidate_prefers_last_fenced_block():
    from my_coding_agent.evals.judge import _extract_json_candidate

    content = 'intro ```json\n{"a": 1}\n``` middle text ```json\n{"a": 2}\n``` trailing'
    assert json.loads(_extract_json_candidate(content)) == {"a": 2}


def test_extract_json_candidate_falls_back_to_outermost_braces():
    from my_coding_agent.evals.judge import _extract_json_candidate

    content = 'here is the answer: {"a": 1, "b": {"c": 2}} thanks'
    assert json.loads(_extract_json_candidate(content)) == {"a": 1, "b": {"c": 2}}


def test_extract_json_candidate_no_json_raises():
    from my_coding_agent.evals.judge import _extract_json_candidate

    with pytest.raises(JudgeError, match="no JSON object"):
        _extract_json_candidate("no json here at all")


# --- calibration -----------------------------------------------------------


def test_cohens_kappa_perfect_agreement():
    assert cohens_kappa([1, 2, 3, 1], [1, 2, 3, 1]) == 1.0


def test_cohens_kappa_chance_level_agreement_near_zero():
    # Half-and-half labels with judge scores uncorrelated with human labels.
    a = [1, 1, 2, 2, 1, 1, 2, 2]
    b = [1, 2, 1, 2, 1, 2, 1, 2]
    kappa = cohens_kappa(a, b)
    assert -0.5 < kappa < 0.5


def test_cohens_kappa_mismatched_lengths_raises():
    with pytest.raises(ValueError, match="same length"):
        cohens_kappa([1, 2], [1])


def test_cohens_kappa_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        cohens_kappa([], [])


def test_calibrate_agreeing_judge_reports_high_agreement(mocker, tmp_path):
    rubric = load_rubric(_write_rubric(tmp_path, _well_formed_rubric_data()))
    labelled = [
        LabelledCase(task="q1", output="a1", human_score=5),
        LabelledCase(task="q2", output="a2", human_score=1),
        LabelledCase(task="q3", output="a3", human_score=5),
        LabelledCase(task="q4", output="a4", human_score=1),
    ]
    responses = [
        _judge_json_response({"correctness": 5, "tone": 5}),
        _judge_json_response({"correctness": 1, "tone": 1}),
        _judge_json_response({"correctness": 5, "tone": 5}),
        _judge_json_response({"correctness": 1, "tone": 1}),
    ]
    fake_llm = mocker.Mock()
    fake_llm.chat_completion.side_effect = [
        mocker.Mock(json=lambda r=r: {"choices": [{"message": {"content": r}}]})
        for r in responses
    ]

    result = calibrate(fake_llm, rubric, labelled, criterion_name="correctness")

    assert result.agreement == 1.0
    assert result.reliable is True
    assert result.judge_scores == (5, 1, 5, 1)
    assert result.human_scores == (5, 1, 5, 1)


def test_calibrate_disagreeing_judge_flagged_unreliable(mocker, tmp_path):
    rubric = load_rubric(_write_rubric(tmp_path, _well_formed_rubric_data()))
    labelled = [
        LabelledCase(task="q1", output="a1", human_score=5),
        LabelledCase(task="q2", output="a2", human_score=1),
        LabelledCase(task="q3", output="a3", human_score=5),
        LabelledCase(task="q4", output="a4", human_score=1),
    ]
    # Judge disagrees with every human label.
    responses = [
        _judge_json_response({"correctness": 1, "tone": 1}),
        _judge_json_response({"correctness": 5, "tone": 5}),
        _judge_json_response({"correctness": 1, "tone": 1}),
        _judge_json_response({"correctness": 5, "tone": 5}),
    ]
    fake_llm = mocker.Mock()
    fake_llm.chat_completion.side_effect = [
        mocker.Mock(json=lambda r=r: {"choices": [{"message": {"content": r}}]})
        for r in responses
    ]

    result = calibrate(
        fake_llm, rubric, labelled, criterion_name="correctness", threshold=0.6
    )

    assert result.reliable is False
    assert result.agreement < 0.6


def test_calibrate_unknown_criterion_raises(mocker, tmp_path):
    rubric = load_rubric(_write_rubric(tmp_path, _well_formed_rubric_data()))
    labelled = [LabelledCase(task="q", output="a", human_score=5)]
    fake_llm = _fake_llm(mocker, _judge_json_response({"correctness": 5, "tone": 5}))
    with pytest.raises(JudgeError, match="no criterion"):
        calibrate(fake_llm, rubric, labelled, criterion_name="nonexistent")


# --- JudgeScorer (eval harness Scorer contract) ----------------------------


def _run_result(output, errored=False):
    return RunResult(
        final_output=output,
        session_dir=Path("/tmp/session"),
        session_id="sess1",
        errored=errored,
    )


def test_judge_scorer_registered_under_judge_ref():
    assert isinstance(resolve_scorer("judge"), JudgeScorer)


def test_judge_scorer_passes_when_overall_score_meets_threshold(mocker, tmp_path):
    rubric_path = _write_rubric(tmp_path, _well_formed_rubric_data())
    fake_llm = _fake_llm(
        mocker, _judge_json_response({"correctness": 5, "tone": 5}, overall_score=4.0)
    )
    scorer = JudgeScorer(llm=fake_llm)
    case = EvalCase(
        id="c1",
        task="2+2?",
        scorer="judge",
        expected={"rubric": str(rubric_path), "pass_threshold": 3.5},
    )

    score = scorer.score(case, _run_result("4"))

    assert score.passed is True
    assert score.metrics["overall_score"] == 4.0
    assert score.metrics["correctness_score"] == 5.0
    assert "overall_rationale" in score.detail


def test_judge_scorer_fails_when_overall_score_below_threshold(mocker, tmp_path):
    rubric_path = _write_rubric(tmp_path, _well_formed_rubric_data())
    fake_llm = _fake_llm(
        mocker, _judge_json_response({"correctness": 1, "tone": 1}, overall_score=1.0)
    )
    scorer = JudgeScorer(llm=fake_llm)
    case = EvalCase(
        id="c1",
        task="2+2?",
        scorer="judge",
        expected={"rubric": str(rubric_path), "pass_threshold": 3.5},
    )

    score = scorer.score(case, _run_result("banana"))
    assert score.passed is False


def test_judge_scorer_errored_run_fails_without_calling_judge(mocker, tmp_path):
    rubric_path = _write_rubric(tmp_path, _well_formed_rubric_data())
    fake_llm = mocker.Mock()
    scorer = JudgeScorer(llm=fake_llm)
    case = EvalCase(
        id="c1",
        task="q",
        scorer="judge",
        expected={"rubric": str(rubric_path), "pass_threshold": 3.5},
    )

    score = scorer.score(case, _run_result("", errored=True))

    assert score.passed is False
    assert score.detail["reason"] == "agent run errored"
    fake_llm.chat_completion.assert_not_called()


def test_judge_scorer_missing_rubric_field_fails_clearly(mocker):
    scorer = JudgeScorer(llm=mocker.Mock())
    case = EvalCase(id="c1", task="q", scorer="judge", expected={"pass_threshold": 3})
    score = scorer.score(case, _run_result("a"))
    assert score.passed is False
    assert "rubric" in score.detail["reason"]


def test_judge_scorer_missing_threshold_field_fails_clearly(mocker, tmp_path):
    rubric_path = _write_rubric(tmp_path, _well_formed_rubric_data())
    scorer = JudgeScorer(llm=mocker.Mock())
    case = EvalCase(
        id="c1", task="q", scorer="judge", expected={"rubric": str(rubric_path)}
    )
    score = scorer.score(case, _run_result("a"))
    assert score.passed is False
    assert "pass_threshold" in score.detail["reason"]


def test_judge_scorer_malformed_rubric_fails_clearly(mocker, tmp_path):
    bad_path = tmp_path / "bad.json"
    bad_path.write_text("not json")
    scorer = JudgeScorer(llm=mocker.Mock())
    case = EvalCase(
        id="c1",
        task="q",
        scorer="judge",
        expected={"rubric": str(bad_path), "pass_threshold": 3},
    )
    score = scorer.score(case, _run_result("a"))
    assert score.passed is False
    assert "not valid JSON" in score.detail["reason"]


# --- inline rubric ----------------------------------------------------------


def test_judge_scorer_inline_rubric_matches_path_form(mocker, tmp_path):
    rubric_data = _well_formed_rubric_data()
    rubric_path = _write_rubric(tmp_path, rubric_data)
    response = _judge_json_response({"correctness": 5, "tone": 5}, overall_score=4.0)

    path_scorer = JudgeScorer(llm=_fake_llm(mocker, response))
    path_case = EvalCase(
        id="c1",
        task="2+2?",
        scorer="judge",
        expected={"rubric": str(rubric_path), "pass_threshold": 3.5},
    )
    path_score = path_scorer.score(path_case, _run_result("4"))

    inline_scorer = JudgeScorer(llm=_fake_llm(mocker, response))
    inline_case = EvalCase(
        id="c1",
        task="2+2?",
        scorer="judge",
        expected={"rubric": rubric_data, "pass_threshold": 3.5},
    )
    inline_score = inline_scorer.score(inline_case, _run_result("4"))

    assert inline_score.passed == path_score.passed
    assert inline_score.metrics == path_score.metrics
    assert inline_score.detail == path_score.detail


def test_judge_scorer_malformed_inline_rubric_fails_clearly(mocker):
    scorer = JudgeScorer(llm=mocker.Mock())
    data = _well_formed_rubric_data()
    del data["scale"]
    case = EvalCase(
        id="c1",
        task="q",
        scorer="judge",
        expected={"rubric": data, "pass_threshold": 3},
    )

    score = scorer.score(case, _run_result("a"))

    assert score.passed is False
    assert "missing required field" in score.detail["reason"]
    scorer._llm.chat_completion.assert_not_called()
