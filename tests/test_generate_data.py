from training.generate_data import generate_dataset


def test_generation_is_reproducible_and_answers_group_examples():
    answers = ["cigar", "rebut", "sissy", "humph", "awake", "blush"]
    allowed = answers + ["adieu", "arise", "soare"]
    first = generate_dataset(answers, allowed, states=4, seed=14)
    second = generate_dataset(answers, allowed, states=4, seed=14)
    assert (first["features"] == second["features"]).all()
    assert (first["targets"] == second["targets"]).all()
    assert (first["answer_ids"] == second["answer_ids"]).all()


def test_empty_lists_fail_clearly():
    import pytest
    with pytest.raises(ValueError, match="must not be empty"):
        generate_dataset([], ["adieu"], states=1)
