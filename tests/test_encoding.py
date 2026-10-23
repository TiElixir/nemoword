from wordle.encoding import FEATURE_SIZE, encode_state_action


def test_fixed_feature_size_and_hidden_answer_independent_inputs():
    first = encode_state_action([], ["cigar", "rebut"], "adieu", total_answers=2)
    second = encode_state_action([], ["cigar", "rebut"], "adieu", total_answers=2)
    assert len(first) == FEATURE_SIZE
    assert first == second


def test_encodes_observed_history():
    features = encode_state_action([("cigar", (2, 0, 1, 0, 2))], ["rebut"], "rebut", total_answers=2)
    assert len(features) == FEATURE_SIZE
    assert any(features)
