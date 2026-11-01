from src.repetition_code import (
    encode,
    receive_and_interpret,
    get_code_error_probability,
)

def test_encode_structure():
    """Verify repetition code encoding lengths and values."""
    assert encode(0, 3) == [0, 0, 0]
    assert encode(1, 5) == [1, 1, 1, 1, 1]

def test_majority_vote_decoding():
    """Verify that majority voting correctly identifies the dominant bit."""
    # Bit 0 majority
    assert receive_and_interpret([0, 1, 0], majority_vote=True) == 0
    # Bit 1 majority
    assert receive_and_interpret([1, 1, 0, 0, 1], majority_vote=True) == 1
    # Frequency counts when majority_vote=False
    counts = receive_and_interpret([1, 1, 0], majority_vote=False)
    assert counts[0] == (1, 2)  # bit 1 appears 2 times

def test_error_suppression_below_threshold():
    """
    At low physical error rate (p = 0.05), increasing distance from
    d=3 to d=5 must reduce the logical error rate.
    """
    err_d3 = get_code_error_probability(code_distance=3, p_error=0.05, n_shots=4000)
    err_d5 = get_code_error_probability(code_distance=5, p_error=0.05, n_shots=4000)
    assert err_d5 < err_d3
