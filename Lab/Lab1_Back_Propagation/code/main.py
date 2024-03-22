import model, data
import json


def main() -> None:
    # Read config
    print("Read config...")
    with open("Lab/Lab1_Back_Propagation/code/config.json", "r") as f:
        config = json.load(f)
    print("config:", config)

    # Generate data
    print("Generate data...")
    if config["data_type"] == "linear":
        inputs, labels = data.generate_linear()
    else:
        inputs, labels = data.generate_xor_easy()

    # Create neural network
    print("Create neural network...")
    neural_network = model.NeuralNetwork(
        epoch=config["epoch"],
        learning_rate=config["learning_rate"],
        hidden_units=config["hidden_units"],
        activation=config["activation"],
        optimizer=config["optimizer"]
    )
    
    # Train neural network
    print("Train neural network...")
    neural_network.train(inputs=inputs, labels=labels)
    
    # Test neural network
    print("Test neural network...")
    neural_network.test(inputs=inputs, labels=labels)


if __name__ == '__main__':
    print("-------------------------------------")
    print("Start!")
    main()
