import pickle
import gzip
import random
import numpy as np
import matplotlib.pyplot as plt

from network import Network
import logic

def load_mnist_dataset(filepath='mnist.pkl.gz'):
    """
    Loads and unpacks the MNIST dataset.
    Returns: (training_data, validation_data, test_data) or None if fails.
    """
    try:
        with gzip.open(filepath, 'rb') as f:
            return pickle.load(f, encoding="latin1")
    except FileNotFoundError:
        print(f"Error: Dataset file '{filepath}' not found.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during data loading: {e}")
        return None

def prepare_data():
    """
    Reshapes raw data into vectors suitable for the neural network.
    """
    raw_data = load_mnist_dataset()
    if raw_data is None:
        return None, None
    
    train_set, _, test_set = raw_data
    
    # Process training data
    train_inputs = [np.reshape(x, (784, 1)) for x in train_set[0]]
    train_labels = [np.eye(10)[y].reshape(10, 1) for y in train_set[1]]
    training_data = list(zip(train_inputs, train_labels))
    
    # Process test data
    test_inputs = [np.reshape(x, (784, 1)) for x in test_set[0]]
    test_data = list(zip(test_inputs, test_set[1]))
    
    return training_data, test_data

def execute_neuro_symbolic_inference(model, test_data, mode="prime"):
    """
    Implements the handshake between neural intuition and symbolic rules.
    """
    # Select random sample
    idx = random.randint(0, len(test_data) - 1)
    image, actual_label = test_data[idx]
    
    print(f"\n--- Inference Report (Index {idx}) ---")
    print(f"Ground Truth: {actual_label}")
    
    # System-1: Neural Network Rankings
    predictions = logic.get_ai_rankings(model, image)
    top_guess, confidence = predictions[0]
    print(f"Initial AI Prediction: {top_guess} ({confidence*100:.2f}% confidence)")
    
    # System-2: Symbolic Constraint Verification
    final_selection = None
    for digit, _ in predictions:
        # Check constraints
        valid = logic.is_prime_logic(digit) if mode == "prime" else logic.is_even_logic(digit)
        
        if valid:
            final_selection = digit
            break
            
    # Result Analysis
    if final_selection == top_guess:
        print(f"Validation: Prediction consistent with '{mode}' constraint.")
    else:
        print(f"Correction: Logic layer selected {final_selection} over {top_guess} to satisfy '{mode}' rule.")

    # Visualization
    plt.imshow(image.reshape(28, 28), cmap='gray')
    plt.title(f"Actual: {actual_label} | Logic Decision: {final_selection}")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    # Data Preparation
    training_data, test_data = prepare_data()

    if training_data and test_data:
        # Initialize Architecture
        mnist_net = Network([784, 30, 10])

        # Training Phase
        print("Commencing network training...")
        mnist_net.SGD(training_data, epochs=5, mini_batch_size=10, eta=3.0, test_data=test_data)

        # Inference Phase
        execute_neuro_symbolic_inference(mnist_net, test_data, mode="prime")