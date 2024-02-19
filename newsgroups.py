from typing import List, Dict
import torch
from torch.nn.utils.rnn import pad_sequence

from data_utils import DataUtils
from vocabulary import Vocabulary


class NewsGroupsDataset:
    def __init__(
        self,
        train_data_filename: str,
        train_labels_filename: str,
        dev_data_filename: str,
        dev_labels_filename: str,
        test_data_filename: str,
    ):
        # Load the data and labels
        self._load_data_and_label(
            train_data_filename,
            train_labels_filename,
            dev_data_filename,
            dev_labels_filename,
            test_data_filename,
        )
        # data: a list of texts
        # labels: a list of labels (converted to integers)

    def featurize(self, model_type: str, feature_type: str, feature_config: Dict):
        self.feature_type = feature_type
        self.model_type = model_type
        self.feature_config = feature_config

        if model_type == "perceptron":
            train_features, val_features, dev_features, test_features = (
                self._featurize_perceptron()
            )

        elif model_type == "mlp":
            train_features, val_features, dev_features, test_features = (
                self._featurize_mlp()
            )

            return (
                (
                    (train_features, torch.tensor(self.train_labels)),
                    (val_features, torch.tensor(self.val_labels)),
                    (dev_features, torch.tensor(self.dev_labels)),
                    (test_features, torch.tensor(self.test_labels)),
                ),
                (
                    self.label_to_index,
                    len(self.vocab),
                ),
            )

    def _featurize_mlp(self):
        # 1. Build the vocabulary from the training data
        vocab = Vocabulary(self.feature_config)
        vocab.build_vocabulary_from_data(self.train_data)
        print(f">> Vocabulary size: {len(vocab)}")
        self.vocab = vocab

        # 2. Encode the data by first tokenizing and then converting to indices
        train_features = [vocab.encode(sentence) for sentence in self.train_data]
        val_features = [vocab.encode(sentence) for sentence in self.val_data]
        dev_features = [vocab.encode(sentence) for sentence in self.dev_data]
        test_features = [vocab.encode(sentence) for sentence in self.test_data]

        # 3. Pad the sequences
        train_features = pad_sequence(
            [torch.tensor(f) for f in train_features], batch_first=True
        )
        val_features = pad_sequence(
            [torch.tensor(f) for f in val_features], batch_first=True
        )
        dev_features = pad_sequence(
            [torch.tensor(f) for f in dev_features], batch_first=True
        )
        test_features = pad_sequence(
            [torch.tensor(f) for f in test_features], batch_first=True
        )

        return train_features, val_features, dev_features, test_features

    def _featurize_perceptron(self):
        pass

    def _load_data_and_label(
        self,
        train_data_filename: str,
        train_labels_filename: str,
        dev_data_filename: str,
        dev_labels_filename: str,
        test_data_filename: str,
    ):
        # Load the sentences and labels from CSV files
        train_data, train_labels = DataUtils.load_csv(
            data_filename=train_data_filename, labels_filename=train_labels_filename
        )
        dev_data, dev_labels = DataUtils.load_csv(
            data_filename=dev_data_filename, labels_filename=dev_labels_filename
        )
        test_data, test_labels = DataUtils.load_csv(
            data_filename=test_data_filename, labels_filename="", test=True
        )

        # Create the mapping from label to index
        label_to_index = {}
        for label in train_labels:
            if label not in label_to_index:
                label_to_index[label] = len(label_to_index)
        self.label_to_index = label_to_index
        self.index_to_label = {v: k for k, v in label_to_index.items()}

        # Convert the labels to indices
        train_labels = [label_to_index[label] for label in train_labels]
        dev_labels = [label_to_index[label] for label in dev_labels]

        # Split the train data into train and validation data
        train_data, train_labels, val_data, val_labels = DataUtils.train_val_split(
            train_data, train_labels
        )

        # Print the size of the data
        print(f">> Train data size: \t {len(train_data)}")
        print(f">> Val data size: \t {len(val_data)}")
        print(f">> Dev data size: \t {len(dev_data)}")
        print(f">> Test data size: \t {len(test_data)}")

        # Update the data
        self.train_data = train_data
        self.train_labels = train_labels
        self.val_data = val_data
        self.val_labels = val_labels
        self.dev_data = dev_data
        self.dev_labels = dev_labels
        self.test_data = test_data
        self.test_labels = test_labels


def newsgroups_data_loader(
    train_data_filename: str,
    train_labels_filename: str,
    dev_data_filename: str,
    dev_labels_filename: str,
    test_data_filename: str,
    feature_type: str,
    model_type: str,
    feature_config: Dict = {},
):
    """Loads the data.

    Inputs:
        train_data_filename: The filename of the training data.
        train_labels_filename: The filename of the training labels.
        dev_data_filename: The filename of the development data.
        dev_labels_filename: The filename of the development labels.
        test_data_filename: The filename of the test data.
        feature_type: The type of features to use.
        model_type: The type of model to use.

    Returns:
        Training, validation, dev, and test data, all represented as a list of (input, label) format.

        Suggested: for test data, put in some dummy value as the label.
    """
    # TODO: Load the data from the text format.
    print(f"> Loading data from files...")
    dataset = NewsGroupsDataset(
        train_data_filename,
        train_labels_filename,
        dev_data_filename,
        dev_labels_filename,
        test_data_filename,
    )

    # TODO: Featurize the input data for all three splits.
    print(f"> Featurizing the data...")
    return dataset.featurize(model_type, feature_type, feature_config)
