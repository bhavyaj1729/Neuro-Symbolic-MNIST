import numpy as np

def is_prime_logic(n):
    """Formal rule: Returns True if n is prime."""
    if n < 2: return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0: return False
    return True

def is_even_logic(n):
    """Formal rule: Returns True if n is even."""
    return n % 2 == 0

def get_ai_rankings(network, image):
    """Get all 10 digit probabilities from the neural net, sorted by confidence."""
    probabilities = network.feedforward(image).flatten()
    rankings = list(enumerate(probabilities))
    rankings.sort(key=lambda x: x[1], reverse=True)
    return rankings