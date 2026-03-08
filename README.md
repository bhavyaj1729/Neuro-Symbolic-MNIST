Most modern AI is "System-1"—it’s fast, intuitive, and great at recognizing patterns, but it doesn't truly reason. When a standard Neural Network sees a messy handwritten digit, it makes a statistical guess. If that guess is mathematically impossible, the network usually doesn't know any better.

This project implements a Neuro-Symbolic architecture to bridge that gap. By combining a raw Neural Network (System-1) with a Formal Logic layer (System-2), we create a system that doesn't just "see" numbers—it understands the rules they must follow.

 
 Why this is different?

Instead of relying solely on a "black box" model, this pipeline uses a Logical Handshake:

The Brain (Neural Net): Proposes a ranked list of what it thinks the digit is based on visual patterns.

The Critic (Symbolic Logic): Checks those guesses against hard mathematical rules (like Prime or Even number constraints).

The Decision: If the AI's top guess violates the rules, the Logic layer vetoes it and selects the next best visually similar candidate that is logically valid.


📂 Project Structure


I've organized the code into three distinct modules to follow professional software engineering standards:

network.py: My "from-scratch" Neural Network. No high-level libraries here—just pure NumPy, Backpropagation, and Stochastic Gradient Descent.


logic.py: The symbolic engine. This is where the mathematical "laws" live.


main.py: The orchestrator that handles data loading and manages the interaction between the neurons and the logic.

What's Next?

Currently, the Logic layer acts as a filter. It catches mistakes, but it doesn't "teach" the brain.

My next goal is Logical Training. I want to implement a feedback loop where, if the Logic layer catches a contradiction, it sends a signal back through the Neural Network to update its weights. This would move the system from just "filtering" errors to actually learning from logical failures.
