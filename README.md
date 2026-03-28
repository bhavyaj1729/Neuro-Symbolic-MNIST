# Neuro-Symbolic MNIST

A from-scratch implementation of a neuro-symbolic inference system applied to the MNIST handwritten digit dataset. The project combines a custom-built neural network with a symbolic logic layer to demonstrate how structured reasoning can guide and correct the output of a neural classifier.

---

## What This Project Does

Standard neural networks output a prediction based on the highest probability class. This project goes a step further: after the network produces its ranked predictions, a symbolic logic layer intercepts the result and enforces a formal constraint — such as "the digit must be prime" or "the digit must be even."

If the top neural prediction satisfies the constraint, it is accepted. If it does not, the symbolic layer walks down the ranked predictions and selects the highest-confidence digit that does satisfy the rule. This models the interaction between fast, intuition-based recognition (the neural network) and slow, rule-based verification (the symbolic layer) — analogous to System 1 and System 2 thinking in cognitive science.

---

## How It Works

The pipeline has three stages.

**Stage 1 — Neural Perception**

A fully-connected neural network with the architecture `[784 -> 30 -> 10]` is trained on MNIST using stochastic gradient descent and backpropagation. All weights and biases are implemented from scratch using NumPy. The network takes a flattened 28x28 pixel image as input and outputs a 10-dimensional probability vector, one value per digit class (0-9).

**Stage 2 — Ranked Candidate Generation**

Rather than taking only the top prediction, all 10 digit probabilities are ranked in descending order of confidence. This ranked list is passed to the symbolic module as a set of candidates.

**Stage 3 — Symbolic Constraint Verification**

The symbolic module iterates through the ranked candidates and applies a logical rule to each one:

- In `prime` mode, it selects the first candidate digit that is a prime number (2, 3, 5, 7).
- In `even` mode, it selects the first candidate digit that is even (0, 2, 4, 6, 8).

The first candidate that satisfies the rule becomes the final output. If the neural network's top prediction already satisfies the constraint, no correction is made. If it does not, the system corrects itself using the next best valid candidate.

---

## Project Structure

```
Neuro-Symbolic-MNIST/
|
|-- network.py       # Neural network built from scratch (feedforward, SGD, backpropagation)
|-- logic.py         # Symbolic rules (is_prime, is_even) and ranked prediction utility
|-- main.py          # Data loading, training, and neuro-symbolic inference pipeline
|-- mnist.pkl.gz     # MNIST dataset file (must be present locally)
```

---

## Neural Network Architecture

The network is implemented entirely in NumPy with no deep learning frameworks.

| Component        | Detail                                          |
|------------------|-------------------------------------------------|
| Input layer      | 784 neurons (28x28 flattened pixel values)      |
| Hidden layer     | 30 neurons                                      |
| Output layer     | 10 neurons (one per digit class, 0-9)           |
| Activation       | Sigmoid                                         |
| Loss             | Mean squared error (via delta in backpropagation)|
| Optimizer        | Stochastic gradient descent (SGD)               |
| Default training | 5 epochs, mini-batch size 10, learning rate 3.0 |

Weights are initialized using a standard normal distribution. The backpropagation algorithm computes gradients layer by layer using the chain rule and updates weights and biases at the end of each mini-batch.

---

## Symbolic Logic Module

The symbolic rules in `logic.py` are deterministic and human-readable.

| Function            | Description                                                         |
|---------------------|---------------------------------------------------------------------|
| `is_prime_logic(n)` | Returns True if n is a prime number. Used in `prime` mode.          |
| `is_even_logic(n)`  | Returns True if n is even. Used in `even` mode.                     |
| `get_ai_rankings()` | Returns all 10 digit predictions sorted by confidence, descending.  |

The rules act as hard constraints. The symbolic layer does not modify the neural network's weights or training in any way — it operates purely at inference time, after the network has already produced its output.

---

## Installation

**Requirements**

- Python 3.8 or higher
- NumPy
- Matplotlib

**Setup**

```bash
# Clone the repository
git clone https://github.com/bhavyaj1729/Neuro-Symbolic-MNIST.git
cd Neuro-Symbolic-MNIST

# Install dependencies
pip install numpy matplotlib
```

The MNIST dataset (`mnist.pkl.gz`) must be present in the project root. It can be downloaded from:
http://deeplearning.net/data/mnist/mnist.pkl.gz

---

## Usage

Run the full pipeline — data loading, training, and neuro-symbolic inference — with:

```bash
python main.py
```

This will:
1. Load and preprocess the MNIST dataset
2. Train the neural network for 5 epochs, printing accuracy after each epoch
3. Select a random test image and run neuro-symbolic inference in `prime` mode by default
4. Display the image with the ground truth label and the logic-corrected prediction

To switch between constraint modes, change the `mode` argument at the bottom of `main.py`:

```python
# Use prime number constraint (default)
execute_neuro_symbolic_inference(mnist_net, test_data, mode="prime")

# Use even number constraint
execute_neuro_symbolic_inference(mnist_net, test_data, mode="even")
```

---

## Sample Inference Output

```
--- Inference Report (Index 4821) ---
Ground Truth: 3
Initial AI Prediction: 1 (72.14% confidence)
Correction: Logic layer selected 3 over 1 to satisfy 'prime' rule.
```

In this example, the neural network was most confident about digit `1`. Since `1` is not a prime number, the symbolic layer overrode it and selected `3` — the next highest-confidence candidate that satisfies the primality constraint — which also happens to match the ground truth label.

---

## Key Concepts Demonstrated

**Neuro-symbolic integration at inference time**
The symbolic layer does not retrain or modify the network. It operates as a post-hoc filter on an already-trained model's output, making the approach lightweight and decoupled from training.

**Ranked prediction as a candidate pool**
Instead of committing to a single hard prediction, the system treats the neural output as a ranked list of hypotheses. The symbolic layer reasons over this full list rather than making a binary accept-or-reject decision on just the top result.

**Interpretable corrections**
Every correction made by the symbolic layer is fully explainable: which digit was originally predicted, which digit was ultimately selected, and which rule caused the switch.

**Separation of concerns**
The neural module handles perception — recognizing patterns in pixel data. The symbolic module handles reasoning — applying domain rules. Each component can be modified or extended independently.

---

## Planned Improvements

- Evaluate the constraint satisfaction rate and logic correction rate across the full test set
- Support user-defined symbolic rules without modifying source code
- Extend to multi-digit reasoning tasks, such as verifying that two digit images sum to a prime
- Add a visualization of the full confidence ranking alongside the symbolic decision for each inference run

---

## References

- LeCun, Y., Cortes, C., and Burges, C.J.C. — The MNIST Database of Handwritten Digits.
  http://yann.lecun.com/exdb/mnist/
- Nielsen, M. — Neural Networks and Deep Learning.
  http://neuralnetworksanddeeplearning.com/
  (The network architecture and SGD implementation follow Nielsen's pedagogical approach.)
- Kahneman, D. — Thinking, Fast and Slow. Farrar, Straus and Giroux, 2011.
  (Conceptual basis for the System 1 / System 2 framing used in this project.)

---

## License

This project is licensed under the MIT License.
