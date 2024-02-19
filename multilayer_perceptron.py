""" Multi-layer perceptron model for Assignment 1: Starter code.

You can change this code while keeping the function headers.
"""

import os
import argparse
import json

from util import evaluate, load_data
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class MultilayerPerceptronModel(nn.Module):
    """Multi-layer perceptron model for classification."""

    def __init__(
        self,
        num_classes,
        vocab_size,
        embedding_dim=128,
        hidden_dim=[256, 128],
        num_layers=2,
        activation="relu",
        label_to_index=None,
    ):
        """Initializes the model.

        Inputs:
            num_classes (int): The number of classes.
            vocab_size (int): The size of the vocabulary.
        """
        # TODO: Implement initialization of this model.
        # Note: You can add new arguments, with a default value specified.
        super(MultilayerPerceptronModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(embedding_dim, hidden_dim[0]))

        # Dynamically add hidden layers based on num_layers
        for i in range(1, num_layers):
            self.layers.append(nn.Linear(hidden_dim[i - 1], hidden_dim[i]))

        # Final layer
        self.layers.append(nn.Linear(hidden_dim[-1], num_classes))

        # Activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        else:
            raise ValueError("Activation function not supported")

        # Softmax function
        self.softmax = nn.Softmax(dim=1)

        # Create label to index and index to label mappings
        if label_to_index:
            self.label2index = label_to_index
            self.index2label = {v: k for k, v in label_to_index.items()}
        else:
            self.label2index = {}
            self.index2label = {}

        print(label_to_index)

    def label_to_index(self, label):
        return self.label2index[label]

    def index_to_label(self, index):
        return self.index2label[index]

    def forward(self, x):
        x = self.embeddings(x)
        x = torch.mean(x, dim=1)  # Average embedding values across the sequence length

        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.activation(x)

        x = self.layers[-1](x)
        # x = self.softmax(x)
        return x

    def predict(self, model_input: torch.Tensor, evaluate=False):
        """Predicts a label for an input.

        Inputs:
            model_input (tensor): Input data for an example or a batch of examples.

        Returns:
            The predicted class.

        """
        # TODO: Implement prediction for an input.
        # If the input just have one dimension, add a dimension to it in the first axis
        if len(model_input.shape) == 1:
            model_input = model_input.unsqueeze(0)

        if not evaluate:
            with torch.no_grad():
                output = self.forward(model_input)
            return torch.argmax(output, dim=1)
        else:
            output = self.forward(model_input)
            label_index = int(torch.argmax(output, dim=1))
            return self.index_to_label(label_index)

    def learn(
        self,
        training_data,
        val_data,
        loss_fct,
        optimizer,
        num_epochs,
        lr,
        patience=5,
        device="cpu",
    ) -> None:
        """Trains the MLP.

        Inputs:
            training_data: Suggested type for an individual training example is
                an (input, label) pair or (input, id, label) tuple.
                You can also use a dataloader.
            val_data: Validation data.
            loss_fct: The loss function.
            optimizer: The optimization method.
            num_epochs: The number of training epochs.
        """
        # TODO: Implement the training of this model.
        best_val_loss = float("inf")
        epochs_no_improve = 0
        optimizer = optimizer(self.parameters(), lr=lr)

        for epoch in range(num_epochs):
            self.train()  # Set the model to training mode
            for inputs, labels in training_data:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()  # Zero the gradients
                outputs = self.forward(inputs)
                loss = loss_fct(outputs, labels)
                loss.backward()  # Backpropagate the loss
                optimizer.step()  # Update the weights

            self.eval()  # Set the model to evaluation mode
            with torch.no_grad():
                val_loss = 0
                for inputs, labels in val_data:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = self.forward(inputs)
                    val_loss += loss_fct(outputs, labels)

            val_loss /= len(val_data)

            if patience == -1:
                # Do not use early stopping
                print(
                    f"Epoch {epoch+1}/{num_epochs} \t Loss: {loss.item():.4f} \t Validation Loss: {val_loss:.4f}"
                )
                continue

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                print(
                    f"Epoch {epoch+1}/{num_epochs} \t Loss: {loss.item():.4f} \t Validation Loss: {val_loss:.4f}"
                )
            else:
                epochs_no_improve += 1
                print(
                    f"Epoch {epoch+1}/{num_epochs} \t Loss: {loss.item():.4f} \t Validation Loss: {val_loss:.4f} \t Early stopping check {epochs_no_improve}/{patience}"
                )

            if epochs_no_improve >= patience:
                print("Early stopping at epoch", epoch + 1)
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MultiLayerPerceptron model")
    parser.add_argument(
        "-p",
        "--params",
        type=str,
        required=True,
        help="Path to the JSON file containing hyperparameters",
    )

    args = parser.parse_args()

    with open(args.params) as f:
        params = json.load(f)

    # Extract parameters from the loaded JSON
    data_type = params.get("data_type", "sst2")
    feature_type = params.get("feature_type", "onehot")
    model_type = params.get("model_type", "mlp")

    # training parameters
    training_config = params.get("training_config", {})
    batch_size = training_config.get("batch_size", 64)
    num_epochs = training_config.get("num_epochs", 50)
    lr = training_config.get("lr", 0.001)
    loss_fct = training_config.get("loss_fct", "cross_entropy")
    optimizer = training_config.get("optimizer", "adam")
    patience = training_config.get("patience", 5)

    # model parameters
    model_config = params.get("model_config", {})
    embedding_dim = model_config.get("embedding_dim", 128)
    hidden_dim = model_config.get("hidden_dim", [256, 128])
    num_layers = model_config.get("num_layers", 2)
    activation = model_config.get("activation", "relu")

    # feature parameters
    feature_config = params.get("feature_config", {})

    # Print the arguments
    print(f"-------- {data_type} | {model_type} --------")
    print(f"Training config \t {training_config}")
    print(f"Model config \t {model_config}")
    print(f"Feature config \t {feature_config}")

    # Set the loss function and optimizer
    loss_fct = nn.CrossEntropyLoss() if loss_fct == "cross_entropy" else nn.MSELoss()
    # Set the optimizer to be Adam or AdaGrad
    optimizer = torch.optim.Adam if optimizer == "adam" else torch.optim.Adagrad

    print("----------------------------------------")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"> Using {device} device for training")

    print("----------------------------------------")

    # Load the data
    (train_data, val_data, dev_data, test_data), (label_to_index, vocab_size) = (
        load_data(data_type, feature_type, model_type, feature_config)
    )

    print("----------------------------------------")

    # Convert the tensors to dataloaders
    train_dataset = TensorDataset(train_data[0], train_data[1])
    val_dataset = TensorDataset(val_data[0], val_data[1])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # print(train_data[0].max(), val_data[0].max(), test_data[0].max())

    # Train the model using the training data.
    model = MultilayerPerceptronModel(
        num_classes=2 if data_type == "sst2" else 20,
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        activation=activation,
        label_to_index=label_to_index,
    ).to(device)

    print("Training the model...")
    # Note: ensure you have all the inputs to the arguments.
    model.learn(
        train_loader, val_loader, loss_fct, optimizer, num_epochs, lr, patience, device
    )

    # Predict on the development set.
    # Note: if you used a dataloader for the dev set, you need to adapt the code accordingly.
    # Zip the features and labels together
    dev_data = list(zip(dev_data[0], dev_data[1]))
    dev_accuracy = evaluate(
        model,
        dev_data,
        os.path.join("results", f"mlp_{data_type}_{feature_type}_dev_predictions.csv"),
        device,
    )

    print(f"Dev Accuracy: {dev_accuracy:.4f}")

    # Predict on the test set.
    # Note: We don't provide labels for test, so the returned value from this
    # call shouldn't make sense.
    # Zip the features and labels together
    test_data = list(zip(test_data[0], test_data[1]))
    evaluate(
        model,
        test_data,
        os.path.join("results", f"mlp_{data_type}_{feature_type}_test_predictions.csv"),
        device,
    )
