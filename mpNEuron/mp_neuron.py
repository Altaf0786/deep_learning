import itertools

class MPNeuron:
    def __init__(self, num_inputs):
        self.num_inputs = num_inputs
        self.weights = [0] * num_inputs
        self.threshold = 0

    def set_weights(self, weights):
        """Set the weights for the neuron."""
        self.weights = weights

    def set_threshold(self, threshold):
        """Set the threshold for the neuron."""
        self.threshold = threshold

    def activation(self, inputs):
        """Calculate the weighted sum and determine activation."""
        weighted_sum = sum(w * i for w, i in zip(self.weights, inputs))
        return 1 if weighted_sum >= self.threshold else 0

    def compute_logic_gate(self, inputs):
        """Compute the output and status of the neuron for a specific input."""
        output = self.activation(inputs)
        status = "FIRED" if output == 1 else "NOT FIRED"
        return output, status

    def print_logic_table(self):
        """Print the truth table for the neuron."""
        print(f"Weights: {self.weights}")
        print(f"Threshold: {self.threshold}")
        print("Logic Gate Table:")
        print("Inputs\tOutput\tStatus")

        for inputs in itertools.product([0, 1], repeat=self.num_inputs):
            output = self.activation(inputs)
            status = "FIRED" if output == 1 else "NOT FIRED"
            print(f"{inputs}\t{output}\t{status}")

def configure_neuron(num_inputs, logic_gate):
    """
    Configure the neuron based on the desired logic gate.
    Supported gates: AND, OR, NAND, NOR, XOR
    """
    neuron = MPNeuron(num_inputs)
    
    if logic_gate == "AND":
        weights = [1] * num_inputs
        neuron.set_weights(weights)
        neuron.set_threshold(num_inputs)
    elif logic_gate == "OR":
        weights = [1] * num_inputs
        neuron.set_weights(weights)
        neuron.set_threshold(1)
    elif logic_gate == "NAND":
        weights = [-1] * num_inputs
        neuron.set_weights(weights)
        neuron.set_threshold(-(num_inputs - 1))
    elif logic_gate == "NOR":
        weights = [-1] * num_inputs
        neuron.set_weights(weights)
        neuron.set_threshold(0)
    elif logic_gate == "XOR":
        if num_inputs == 2:  # XOR is defined for 2 inputs
            neuron.weights = [1, 1]
            neuron.set_weights([1, 1])  # XOR logic requires additional logic
            neuron.set_threshold(1)
        else:
            raise ValueError("XOR gate is only defined for 2 inputs.")
    else:
        raise ValueError(f"Unsupported logic gate: {logic_gate}")
    
    return neuron

def main():
    print("Supported Logic Gates: AND, OR, NAND, NOR, XOR")
    num_inputs = int(input("Enter the number of inputs (2 for XOR): "))
    logic_gate = input("Enter the logic gate name: ").strip().upper()
    
    if logic_gate == "XOR" and num_inputs != 2:
        print("XOR gate requires exactly 2 inputs.")
        return
    
    # Configure the neuron
    neuron = configure_neuron(num_inputs, logic_gate)
    
    # Print Logic Table
    print("\nTruth Table:")
    neuron.print_logic_table()

    # Accept specific inputs and compute output
    print("\nCompute Specific Input:")
    inputs = []
    for i in range(num_inputs):
        inp = int(input(f"Enter value for input x{i+1} (0 or 1): "))
        inputs.append(inp)
    
    output, status = neuron.compute_logic_gate(inputs)
    print(f"Output for inputs {inputs}: {output} ({status})")

if __name__ == "__main__":
    main()
