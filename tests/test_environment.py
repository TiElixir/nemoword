import pytest

from wordle.environment import WordleGame, play_game, sample_answers


def test_game_solves_and_filters_candidates():
    game = WordleGame("cigar", ["cigar", "rebut"], ["cigar", "rebut"])
    game.guess("cigar")
    assert game.solved
    assert game.candidates == ["cigar"]
    assert game.finished


def test_answer_need_not_be_all_valid_guesses():
    game = WordleGame("cigar", ["cigar"], ["adieu", "cigar"])
    assert game.guess("adieu")


def test_rejects_disallowed_guess_and_post_game_guess():
    game = WordleGame("cigar", ["cigar"], ["cigar"])
    with pytest.raises(ValueError):
        game.guess("adieu")
    game.guess("cigar")
    with pytest.raises(RuntimeError):
        game.guess("cigar")


def test_strategy_play_and_seeded_sampling_are_reproducible():
    game = play_game("cigar", ["cigar", "rebut"], ["cigar", "rebut"], lambda c, a: c[0])
    assert game.solved
    assert sample_answers(["cigar", "rebut", "sissy"], 2, 9) == sample_answers(["cigar", "rebut", "sissy"], 2, 9)
