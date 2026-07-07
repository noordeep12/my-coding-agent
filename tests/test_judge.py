"""Tests for the rubric-based LLM judge: rubric loading, the judge scorer,
and calibration against human labels."""

import json

import pytest

from my_coding_agent.evals.judge import (
    CriterionVerdict,
    JudgeError,
    JudgeVerdict,
    LabelledCase,
    RubricError,
    calibrate,
    cohens_kappa,
    load_rubric,
    score_with_judge,
)

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


def _fake_llm(mocker, content):
    fake_llm = mocker.Mock()
    fake_llm.chat_completion.return_value = mocker.Mock(
        json=lambda: {"choices": [{"message": {"content": content}}]}
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
    told apart on inspection, not just by a single number."""
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
