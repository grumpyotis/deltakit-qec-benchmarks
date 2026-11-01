"""Chapter 1: Classical Intuition & Repetition Codes.

All code taken from Riverlane's Deltakit Textbook:
https://textbook.riverlane.com/en/latest/notebooks/ch1-intuition/classical-intuition.html

Sections covered:
1. Basic repetition code (encode, noisy channel, majority vote decode).
​2. Numerical simulation measuring logical error rate vs physical error rate.
3. Code distance scaling and demonstrating the classical threshold (p = 0.5).
4. Correlated noise breakdown (simulating error bursts / cosmic rays).
"""

from collections import Counter
import random
import matplotlib.pyplot as plotter
import numpy as np
plotter.rcParams['font.family'] = 'Monospace'
import math

# Section 1: Core Repetition Code Building Blocks

def encode(bit, num_copies):
  """Encodes a single bit by repeating it num_copies times.
  num_copies corresponds to the code distance (d).
  need to Use an odd number of copies (e.g. 3, 5, 7) so that majority voting is
  unambiguous.
  """
  return [bit] * num_copies

def send(bits, p_error = 0.1):
  """Simulates an independent, identically distributed bit-flip channel.
  Each bit independently has a probability `p_error` of flipping:
-   If random.random() <= p_error: bit flips (0 -> 1 or 1 -> 0).
-   Otherwise: bit remains unchanged.
  """
  return [1 ^ bit if random.random() <= p_error else bit for bit in bits]

def receive_and_interpret(bits, majority_vote = False):
  """Interprets received bit strings.
-   If majority_vote=True: returns the single bit that won the majority vote.
-   If majority_vote=False: returns the frequency tally of each bit value.
  """
  received_bits_counted = Counter(bits).most_common()
  if majority_vote:
    return received_bits_counted[0][0]
  else:
    return received_bits_counted

# Section 2: Measuring Logical Error Rates (i.i.d. Noise)

def get_code_error_probability(code_distance, p_error, n_shots = 1000):
  """Runs Monte Carlo trials to estimate the logical failure rate.
  Picks a random bit, encodes it, sends it through the channel,
  and checks whether majority-vote decoding recovers the original message.
  """
  code_errors = 0.0
  for _ in range(n_shots):
    desired_message = random.choice([0, 1])
    encoded_message = encode(desired_message, num_copies=code_distance)
    sent_message = send(encoded_message, p_error=p_error)
    received_message = receive_and_interpret(sent_message, majority_vote=True)
    if received_message != desired_message:
      code_errors += 1

  return code_errors / n_shots

# Section 3: Correlated Noise Model

def flip_all_bits(bits):
  """Flips every bit simultaneously (correlated error burst)."""
  return [1 ^ bit for bit in bits]

def send_with_correlated_bitflip(bits, p_error = 0.1, p_correlated = 1e-4):
  """Simulates noise with correlated error bursts.
-   With probability p_correlated: all bits flip simultaneously.
-   Otherwise: bits flip independently with per-bit probability p_error.
  """
  if random.random() <= p_correlated: 
    # Flip every bit
    return flip_all_bits(bits)
  return send(bits, p_error=p_error)

def get_code_error_probability_correlated(code_distance, p_error, p_correlated, n_shots = 1000):
  """Estimates logical error probability under correlated burst noise."""
  code_errors = 0.0
  for _ in range(n_shots):
    desired_message = random.choice([0, 1])
    encoded_message = encode(desired_message, num_copies=code_distance)
    sent_message = send_with_correlated_bitflip(encoded_message, p_error=p_error, p_correlated=p_correlated)
    received_message = receive_and_interpret(sent_message, majority_vote=True)
    if received_message != desired_message:
      code_errors += 1

  return code_errors / n_shots

# Section 4: Main Execution and Benchmarking

if __name__ == "__main__":
  # sanity check
  msg = encode(1, 5)
  received = receive_and_interpret(send(msg, p_error=0.1), majority_vote=True)
  print(f"Sanity Check -> Encoded: {msg} | Decoded after noise: {received}\n")

  # Simulation 1: Logical Error Rate vs Physical Error Rate
  print("Running Simulation 1: Error suppression across distances...")
  code_distances = [1, 3, 5, 7]
  p_errors = np.logspace(-3, 0, 15)
  n_shots = 5000  # Scaled for speed, use 50k+ for high precision

  results_iid = {}
  for d in code_distances:
    results_iid[d] = [
        get_code_error_probability(d, p, n_shots=n_shots) for p in p_errors
    ]

  plotter.figure(figsize=(7, 4.5))
  for d in code_distances:
    plotter.loglog(p_errors, results_iid[d], marker=".", markersize=8, label=f"d={d}")
  plotter.plot(
      p_errors, p_errors, linestyle="--", color="gray", label="No coding (d=1)"
  )
  plotter.xlabel("Per-bit physical error probability (p)")
  plotter.ylabel("Logical error probability (P_L)")
  plotter.title("Classical Repetition Code: Error Suppression")
  plotter.grid(True, which="both", linestyle=":", alpha=0.6)
  plotter.legend()
  plotter.tight_layout()
  plotter.show()

  # Simulation 2: Demonstrating the Classical Threshold (p = 0.5)
  print("Running Simulation 2: Locating the threshold (p = 0.5)...")
  p_linear = np.linspace(0.1, 0.9, 9)
  threshold_distances = [1, 3, 5, 7, 9]

  results_threshold = {
      d: [get_code_error_probability(d, p, n_shots=3000) for p in p_linear]
      for d in threshold_distances
  }

  plotter.figure(figsize=(7, 4.5))
  for d in threshold_distances:
    plotter.plot(
        p_linear, results_threshold[d], marker="o", markersize=5, label=f"d={d}"
    )
  plotter.axvline(
      0.5,
      color="black",
      linestyle="--",
      label="Threshold p = 0.5",
  )
  plotter.xlabel("Per-bit physical error probability (p)")
  plotter.ylabel("Logical error probability (P_L)")
  plotter.title("Error Correction Threshold: d Scaling Flips at p = 0.5")
  plotter.grid(True, linestyle=":", alpha=0.6)
  plotter.legend()
  plotter.tight_layout()
  plotter.show()

  # Simulation 3: Breakdown Under Correlated Burst Noise
  print("Running Simulation 3: Correlated noise floor comparison...")
  p_burst = np.logspace(-4, 0, 12)
  p_corr_val = 2e-3  # Correlated burst probability
  d_test = [3, 7]

  plotter.figure(figsize=(7, 4.5))
  for d in d_test:
    err_iid = [
        get_code_error_probability(d, p, n_shots=4000) for p in p_burst
    ]
    err_corr = [
        get_code_error_probability_correlated(
            d, p, p_corr_val, n_shots=4000
        )
        for p in p_burst
    ]
    plotter.loglog(p_burst, err_iid, linestyle="-", marker=".", label=f"d={d} i.i.d.")
    plotter.loglog(
        p_burst,
        err_corr,
        linestyle="--",
        marker="s",
        label=f"d={d} with {p_corr_val} burst",
    )

  plotter.axhline(
      p_corr_val,
      color="red",
      linestyle=":",
      label=f"Correlated error floor ({p_corr_val})",
  )
  plotter.xlabel("Per-bit physical error probability (p)")
  plotter.ylabel("Logical error probability (P_L)")
  plotter.title("Impact of Correlated Noise: The Error Floor")
  plotter.grid(True, which="both", linestyle=":", alpha=0.6)
  plotter.legend()
  plotter.tight_layout()
  plotter.show()
