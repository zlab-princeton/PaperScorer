"""Unit test for train_accuracy.py on tokenized data."""
import torch
from transformers import AutoTokenizer
from llamafactory.train.sft.train_accuracy import (
    BoxedFormatHandler,
    TrainAccuracyTracker,
    create_output_format_handler,
)
from llamafactory.extras.constants import IGNORE_INDEX

def test_boxed_format_handler():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", trust_remote_code=True)
    handler = BoxedFormatHandler(tokenizer, positive_token="Accept", negative_token="Reject")

    # Test 1: Tokenize a sample response and verify decision position finding
    response_accept = "Outcome: \\boxed{Accept}"
    response_reject = "Outcome: \\boxed{Reject}"

    tokens_accept = tokenizer.encode(response_accept, add_special_tokens=False)
    tokens_reject = tokenizer.encode(response_reject, add_special_tokens=False)

    print(f"Accept tokens: {tokens_accept}")
    print(f"Decoded: {[tokenizer.decode([t]) for t in tokens_accept]}")
    print(f"Reject tokens: {tokens_reject}")
    print(f"Decoded: {[tokenizer.decode([t]) for t in tokens_reject]}")
    print(f"Brace token ID: {handler.brace_token_id}")
    print(f"Accept token ID: {handler.positive_token_id}")
    print(f"Reject token ID: {handler.negative_token_id}")

    # Test 2: Create input_ids and labels tensors
    # Simulate: input = [prompt tokens... | response tokens], labels = [IGNORE_INDEX... | response tokens]
    prompt = "Will this paper be accepted?"
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)

    # Full sequence for Accept case
    full_tokens_accept = prompt_tokens + tokens_accept
    labels_accept = [IGNORE_INDEX] * len(prompt_tokens) + tokens_accept

    input_ids = torch.tensor([full_tokens_accept])
    labels = torch.tensor([labels_accept])

    # Find decision position - should be position of '{'
    decision_pos = handler.find_decision_positions(input_ids, labels)
    brace_pos_in_response = tokens_accept.index(handler.brace_token_id)
    expected_brace_pos = len(prompt_tokens) + brace_pos_in_response

    print(f"Decision position (should be brace pos): {decision_pos}")
    print(f"Expected brace position: {expected_brace_pos}")

    assert decision_pos[0].item() == expected_brace_pos, f"Expected {expected_brace_pos}, got {decision_pos[0].item()}"
    assert labels[0, decision_pos[0]].item() == handler.brace_token_id, "Decision position should point to brace"
    assert labels[0, decision_pos[0] + 1].item() == handler.positive_token_id, "Position after brace should be Accept"
    print("✓ Decision position correctly finds brace token")

    # Test 3: Verify logits indexing is correct
    # In causal LM: logits[i] predicts token at position i+1
    # So logits[brace_pos] should predict the Accept/Reject token
    batch_size = 2
    seq_len = max(len(full_tokens_accept), len(prompt_tokens) + len(tokens_reject))
    vocab_size = tokenizer.vocab_size

    # Pad sequences to same length
    full_tokens_reject = prompt_tokens + tokens_reject
    labels_reject = [IGNORE_INDEX] * len(prompt_tokens) + tokens_reject

    pad_len_accept = seq_len - len(full_tokens_accept)
    pad_len_reject = seq_len - len(full_tokens_reject)

    input_ids_accept = full_tokens_accept + [tokenizer.pad_token_id or 0] * pad_len_accept
    input_ids_reject = full_tokens_reject + [tokenizer.pad_token_id or 0] * pad_len_reject
    labels_accept_padded = labels_accept + [IGNORE_INDEX] * pad_len_accept
    labels_reject_padded = labels_reject + [IGNORE_INDEX] * pad_len_reject

    input_ids = torch.tensor([input_ids_accept, input_ids_reject])
    labels = torch.tensor([labels_accept_padded, labels_reject_padded])

    # Create logits where model predicts Accept with high prob for sample 0, Reject for sample 1
    logits = torch.zeros(batch_size, seq_len, vocab_size)

    # Key insight: logits[brace_pos] predicts the token at brace_pos+1 (Accept/Reject)
    brace_pos_0 = len(prompt_tokens) + tokens_accept.index(handler.brace_token_id)
    brace_pos_1 = len(prompt_tokens) + tokens_reject.index(handler.brace_token_id)

    print(f"Sample 0: Setting high Accept logit at position {brace_pos_0} (brace position)")
    print(f"Sample 1: Setting high Reject logit at position {brace_pos_1} (brace position)")

    # For sample 0 (Accept ground truth): model predicts Accept
    # Set logits at brace position to predict Accept
    logits[0, brace_pos_0, handler.positive_token_id] = 10.0
    logits[0, brace_pos_0, handler.negative_token_id] = -10.0

    # For sample 1 (Reject ground truth): model predicts Reject
    # Set logits at brace position to predict Reject
    logits[1, brace_pos_1, handler.negative_token_id] = 10.0
    logits[1, brace_pos_1, handler.positive_token_id] = -10.0

    # Compute probabilities
    results = handler.compute_probabilities(logits, input_ids, labels)

    print(f"Predictions: {results['predictions']}")  # Should be [1, 0] (Accept, Reject)
    print(f"Ground truth: {results['ground_truth']}")  # Should be [1, 0]
    print(f"p_positive: {results['p_positive']}")
    print(f"p_negative: {results['p_negative']}")

    assert results['predictions'].tolist() == [1, 0], f"Expected [1, 0], got {results['predictions'].tolist()}"
    assert results['ground_truth'].tolist() == [1, 0], f"Expected [1, 0], got {results['ground_truth'].tolist()}"
    print("✓ Probability computation works correctly with proper logits indexing")

    # Test 4: Verify that wrong logit positions give wrong predictions
    # If we set logits at the WRONG position (Accept position instead of brace position),
    # the predictions should NOT work correctly
    logits_wrong = torch.zeros(batch_size, seq_len, vocab_size)

    # Intentionally set at wrong position (Accept position, not brace position)
    accept_pos_0 = brace_pos_0 + 1  # Wrong! This predicts token at accept_pos+1, not Accept
    logits_wrong[0, accept_pos_0, handler.positive_token_id] = 10.0
    logits_wrong[0, accept_pos_0, handler.negative_token_id] = -10.0

    reject_pos_1 = brace_pos_1 + 1  # Wrong!
    logits_wrong[1, reject_pos_1, handler.negative_token_id] = 10.0
    logits_wrong[1, reject_pos_1, handler.positive_token_id] = -10.0

    results_wrong = handler.compute_probabilities(logits_wrong, input_ids, labels)

    # With wrong positions, predictions should be essentially random (0.5/0.5)
    # since we didn't set any logits at the correct positions
    print(f"Wrong positions - p_positive: {results_wrong['p_positive']}")
    print(f"Wrong positions - p_negative: {results_wrong['p_negative']}")

    # p_positive and p_negative should be roughly equal (no signal)
    p_diff = abs(results_wrong['p_positive'][0] - results_wrong['p_negative'][0])
    assert p_diff < 0.1, f"With wrong logit positions, p_positive and p_negative should be similar, got diff={p_diff}"
    print("✓ Verified that wrong logit positions don't give correct predictions")

    # Test 5: TrainAccuracyTracker
    tracker = TrainAccuracyTracker(handler)
    tracker.update(logits, input_ids, labels, is_training=True)

    metrics = tracker.aggregate_and_reset("train")
    print(f"Aggregated metrics: {metrics}")

    assert "sft_accuracy" in metrics
    assert metrics["sft_accuracy"] == 1.0, f"Expected 100% accuracy, got {metrics['sft_accuracy']}"
    print("✓ TrainAccuracyTracker works correctly")

    print("\n=== All tests passed! ===")

if __name__ == "__main__":
    test_boxed_format_handler()
