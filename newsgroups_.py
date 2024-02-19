from typing import List, Dict
import math
import random
from collections import Counter
import sys
import csv
import torch

from text_preprocessor import TextProcessor


class DataUtils:
    @staticmethod
    def load_csv(data_filename: str, labels_filename: str, test: bool = False):
        """Loads the data from the files."""

        def _load_data_from_file(filename: str):
            """Helper function to load data from a CSV file."""
            # Set the field size limit to the maximum
            # Ref: https://stackoverflow.com/a/15063941
            maxInt = sys.maxsize

            while True:
                # decrease the maxInt value by factor 10
                # as long as the OverflowError occurs.
                try:
                    csv.field_size_limit(maxInt)
                    break
                except OverflowError:
                    maxInt = int(maxInt / 10)

            data = []
            with open(filename, encoding="utf-8") as file:
                reader = csv.reader(file)
                next(reader)  # Skip header
                for row in reader:
                    data.append(row[1])

            return data

        data = _load_data_from_file(data_filename)
        if not test:
            labels = _load_data_from_file(labels_filename)
        else:
            labels = [-1.0 for _ in data]

        return data, labels

    @staticmethod
    def train_val_split(train_data, train_labels, split_ratio=0.8):
        """Splits the data into training and validation sets."""
        # Randomly shuffle the data
        random.seed(0)
        combined = list(zip(train_data, train_labels))
        random.shuffle(combined)
        train_data[:], train_labels[:] = zip(*combined)

        # Split the data
        split_index = int(len(train_data) * split_ratio)
        return (
            train_data[:split_index],
            train_labels[:split_index],
            train_data[split_index:],
            train_labels[split_index:],
        )


class Vocabulary:
    def __init__(self, feature_config: Dict, max_vocab_size=10000) -> None:
        self.itos = {0: "<unk>", 1: "<pad>"}
        self.stoi = {"<unk>": 0, "<pad>": 1}
        self.max_vocab_size = max_vocab_size
        self.feature_config = feature_config

        # Dictionary to store the text processors
        self.text_processors = {
            "preprocess_basic": TextProcessor.basic_preprocess_text,
            "stem_basic": TextProcessor.basic_stem_text,
            "stem_enhanced": TextProcessor.enhanced_stem_text,
        }
        self.text_processor = self.text_processors.get(
            self.feature_config.get("text_processor", "preprocess_basic")
        )

        # Load the stopwords
        if self.feature_config.get("remove_stopwords", True):
            with open("stopwords.txt", "r") as file:
                stopwords = file.read().splitlines()
            self.stopwords = stopwords

        # Initialize the IDF dictionary if the weighting mode is TF-IDF
        if self.feature_config.get("weighting", "binary") == "tfidf":
            self.idf = {}

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def _tokenize(
        self, sentence: str, n: int = 1, remove_stopwords: bool = True
    ) -> List[str]:
        """Tokenize the input text into list of tokens."""
        # Preprocess the text
        text = self.text_processor(sentence)
        # Split the text into tokens
        tokens = text.split()
        # Remove stopwords if specified in the config
        if remove_stopwords:
            tokens = [token for token in tokens if token not in self.stopwords]
        # Generate n-grams if n > 1
        if n > 1:
            tokens = [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]

        return tokens

    def build_vocabulary(self, sentence_list: List[str]) -> None:
        """Builds a vocabulary from a list of raw sentences."""
        if self.feature_config.get("important_ngrams_dict", None):
            important_ngrams_dict = self.feature_config["important_ngrams_dict"]
            for important_token_list in important_ngrams_dict.values():
                for token in important_token_list:
                    self.itos[len(self.itos)] = token
                    self.stoi[token] = len(self.stoi)
            return

        token_list = [
            self._tokenize(
                sentence,
                self.feature_config.get("n", 1),
                self.feature_config.get("remove_stopwords", True),
            )
            for sentence in sentence_list
        ]

        # Filter out the token whose count is less than min_count
        min_count = self.feature_config.get("min_count", 1)
        token_counts = Counter(token for tokens in token_list for token in tokens)
        print(f">>> Number of unique tokens: {len(token_counts)}")
        token_counts = {
            token: count for token, count in token_counts.items() if count >= min_count
        }

        # Build the vocabulary with vocab_size
        vocab_size = min(len(token_counts), self.max_vocab_size - len(self.itos))
        token_counts = dict(
            sorted(token_counts.items(), key=lambda x: x[1], reverse=True)
        )
        for token, count in token_counts.items():
            self.itos[len(self.itos)] = token
            self.stoi[token] = len(self.stoi)
            if len(self.itos) == vocab_size:
                break

        # Calculate the IDF values if the weighting is TF-IDF
        if self.feature_config.get("weighting", "binary") == "tfidf":
            print(f">>> Calculating the IDF values...")
            N = len(sentence_list)
            for token in self.itos.values():
                count = sum([1 for tokens in token_list if token in tokens])
                self.idf[self.stoi[token]] = math.log(N / (count + 1))

    def encode(self, sentence) -> List[int]:
        """Converts a sentence to a list of tokens."""
        # Get the weighting mode
        weighting = self.feature_config.get("weighting", "binary")

        # Build the vector for the important n-grams
        if self.feature_config.get("important_ngrams_dict", None):
            important_ngrams_dict = self.feature_config["important_ngrams_dict"]
            vector = [0] * self.vocab_size
            # Loop through the important n-grams
            for n in important_ngrams_dict.keys():
                tokens = self._tokenize(
                    sentence, n, self.feature_config.get("remove_stopwords", True)
                )
                # Check if each token is in the important n-grams
                for token in tokens:
                    if token in important_ngrams_dict[n]:
                        index = self.stoi.get(token, self.stoi["<unk>"])
                        if weighting == "binary":
                            vector[index] = 1
                        elif weighting == "tf":
                            vector[index] += 1

            # Normalize the vector if the weighting mode is TF
            if weighting == "tf" and sum(vector) > 0:
                norm = math.sqrt(sum([x**2 for x in vector]))
                vector = [x / norm for x in vector]

            return vector

        # Build the vector for the normal n-grams
        tokens = self._tokenize(
            sentence,
            self.feature_config.get("n", 1),
            self.feature_config.get("remove_stopwords", True),
        )
        vector = [0] * (
            self.vocab_size - len(self.feature_config.get("important_bigrams", []))
        )
        for token in tokens:
            index = self.stoi.get(token, self.stoi["<unk>"])
            if weighting == "binary":
                vector[index] = 1
            elif weighting.startswith("tf"):
                vector[index] += 1

        if weighting.startswith("tf") and sum(vector) > 0:
            # TF
            vector = [count / len(tokens) for count in vector]
            if weighting == "tfidf":
                # IDF
                vector = [tf * self.idf[index] for index, tf in enumerate(vector)]
            # Normalize the vector
            norm = math.sqrt(sum([x**2 for x in vector]))
            vector = [x / norm for x in vector]

        return vector


class NewsGroupsDataset:
    def __init__(
        self,
        feature_type: str,
        feature_config: Dict,
        model_type: str,
        max_feature_dim: int = 20000,
    ) -> None:
        self.feature_type = feature_type
        self.feature_config = feature_config
        self.model_type = model_type
        self.max_feature_dim = max_feature_dim

        self.vocab = None

        self.train_data = []
        self.train_labels = []
        self.val_data = []
        self.val_labels = []
        self.dev_data = []
        self.dev_labels = []
        self.test_data = []
        self.test_labels = []

    def load_data(
        self,
        train_data_filename: str,
        train_labels_filename: str,
        dev_data_filename: str,
        dev_labels_filename: str,
        test_data_filename: str,
    ) -> None:
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

        # Map the label to integers
        label_map = {}
        for label in train_labels + dev_labels:
            if label not in label_map:
                label_map[label] = len(label_map)

        train_labels = [label_map[label] for label in train_labels]
        dev_labels = [label_map[label] for label in dev_labels]

        self.label_map = label_map

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

    def featurize(self) -> List:
        if self.model_type == "perceptron":
            train_data, val_data, dev_data, test_data = self._perceptron_featurize()
            return (
                list(zip(train_data, self.train_labels)),
                list(zip(val_data, self.val_labels)),
                list(zip(dev_data, self.dev_labels)),
                list(zip(test_data, self.test_labels)),
                self.label_map,
            )

        elif self.model_type == "mlp":
            train_data, val_data, dev_data, test_data = self._mlp_featurize()
            train_labels = torch.tensor(self.train_labels, dtype=torch.long)
            val_labels = torch.tensor(self.val_labels, dtype=torch.long)
            dev_labels = torch.tensor(self.dev_labels, dtype=torch.long)
            test_labels = torch.tensor(self.test_labels, dtype=torch.long)

            return (
                (train_data, train_labels),
                (val_data, val_labels),
                (dev_data, dev_labels),
                (test_data, test_labels),
                self.label_map,
            )

        else:
            raise ValueError("Invalid model type.")

    def _get_important_ngrams(self, feature_config: Dict) -> List[str]:
        vocab_ = Vocabulary(feature_config, self.max_feature_dim)
        vocab_.build_vocabulary(self.train_data)

        train_dataset = list(zip(self.train_data, self.train_labels))
        encoded_dataset = [
            (vocab_.encode(sentence), label) for sentence, label in train_dataset
        ]

        positive_counts = Counter()
        negative_counts = Counter()
        for encoded_sentence, label in encoded_dataset:
            for i, token_id in enumerate(encoded_sentence):
                if token_id > 0:
                    token = vocab_.itos[i]
                    if label == 1:
                        positive_counts[token] += 1
                    else:
                        negative_counts[token] += 1

        # Remove <unk> and <pad> tokens
        for token in ["<unk>", "<pad>"]:
            if token in positive_counts:
                del positive_counts[token]
            if token in negative_counts:
                del negative_counts[token]

        # Calculate frequencies and their differences
        total_positive = sum(positive_counts.values())
        total_negative = sum(negative_counts.values())

        diffs = {}
        for token in set(positive_counts.keys()).union(negative_counts.keys()):
            positive_freq = (
                positive_counts[token] / total_positive
                if token in positive_counts
                else 0
            )
            negative_freq = (
                negative_counts[token] / total_negative
                if token in negative_counts
                else 0
            )
            diff = abs(positive_freq - negative_freq)
            diffs[token] = diff

        print(f">>> Number of tokens with significant differences: {len(diffs)}")
        # Select the most important bigrams
        first_n = feature_config.get("fea_dim", 256)
        print(f">>> Selecting the top {first_n} important n-grams...")
        return sorted(diffs, key=diffs.get, reverse=True)[:first_n]

    @staticmethod
    def _sentences_to_feature_dicts(
        sentence_list: List[str], vocab: Vocabulary
    ) -> List[Dict[str, int]]:
        """Converts a list of sentences to a list of feature dictionaries."""
        vector_list = [vocab.encode(sentence) for sentence in sentence_list]
        dict_list = []
        for vector in vector_list:
            vector.append(1)  # Bias term
            feature_dict = {i: vector[i] for i in range(len(vector))}
            dict_list.append(feature_dict)
        return dict_list

    def _perceptron_featurize(self) -> List[Dict[int, int]]:
        if self.feature_type == "ngram":
            """
            feature_config = {
                "text_processor": "preprocess_basic" / "stem_basic" / "stem_enhanced",
                "weighting": "binary" / "tfidf",
                "n": 1,
                "min_count": 1,
                "remove_stopwords": True,
            }
            """
            self.vocab = Vocabulary(self.feature_config, self.max_feature_dim)
            self.vocab.build_vocabulary(self.train_data)
            print(f">> Vocabulary size: {self.vocab.vocab_size}")

        elif self.feature_type == "important_ngram":
            """
            feature_config = {
                "text_processor": "preprocess_basic" / "stem_basic" / "stem_enhanced",
                "weighting": "binary" / "tfidf",
                "ns": [1, 2, 3],
                "min_counts": [2, 2, 2],
                "remove_stopwords": True,
                "fea_dims": [2048, 2048, 2048],
            }
            """
            important_ngrams_dict = {}
            for i in range(len(self.feature_config["ns"])):
                # Find the important n-grams
                n = int(self.feature_config["ns"][i])
                print(f">> Finding the important {n}-grams...")
                cur_feature_config = self.feature_config.copy()
                cur_feature_config["n"] = n
                cur_feature_config["min_count"] = self.feature_config["min_counts"][i]
                cur_feature_config["fea_dim"] = self.feature_config["fea_dims"][i]
                important_tokens = self._get_important_ngrams(cur_feature_config)
                important_ngrams_dict[n] = important_tokens
                print(f">>> Important {n}-grams (Top 10): {important_tokens[:10]}")

            new_feature_config = self.feature_config.copy()
            new_feature_config["important_ngrams_dict"] = important_ngrams_dict

            print(f">> Building the vocabulary with the important n-grams...")
            self.vocab = Vocabulary(new_feature_config, self.max_feature_dim)
            self.vocab.build_vocabulary(self.train_data)
            print(f">> Vocabulary size: {self.vocab.vocab_size}")

        else:
            raise ValueError("Invalid feature type.")

        return (
            self._sentences_to_feature_dicts(self.train_data, self.vocab),
            self._sentences_to_feature_dicts(self.val_data, self.vocab),
            self._sentences_to_feature_dicts(self.dev_data, self.vocab),
            self._sentences_to_feature_dicts(self.test_data, self.vocab),
        )

    def _tokens_to_tensors(self, tokens_list, max_length):
        """Converts a list of tokenized text into a tensor."""
        tensor = torch.ones((len(tokens_list), max_length), dtype=torch.long)
        for i, tokens in enumerate(tokens_list):
            for j, token in enumerate(tokens):
                if j >= max_length:
                    break
                tensor[i, j] = self.vocab.stoi.get(token, self.vocab.stoi["<unk>"])

        return tensor

    def _mlp_featurize(self):
        """
        feature_config = {
            "text_processor": "preprocess_basic" / "stem_basic" / "stem_enhanced",
            "min_count": 1,
            "remove_stopwords": True,
        }
        """
        self.vocab = Vocabulary(self.feature_config, self.max_feature_dim * 10)
        self.vocab.build_vocabulary(self.train_data)
        print(f">> Vocabulary size: {self.vocab.vocab_size}")

        train_tokens = [self.vocab._tokenize(text) for text in self.train_data]
        val_tensors = [self.vocab._tokenize(text) for text in self.val_data]
        dev_tensors = [self.vocab._tokenize(text) for text in self.dev_data]
        test_tensors = [self.vocab._tokenize(text) for text in self.test_data]

        # Find the max length of the dataset
        lengths = [len(tokens) for tokens in train_tokens]
        max_length = sorted(lengths)[int(0.9 * len(lengths))]
        # Convert the vectors to tensors
        print("> Converting the data to tensors...")
        train_tensors = self._tokens_to_tensors(train_tokens, max_length)
        val_tensors = self._tokens_to_tensors(val_tensors, max_length)
        dev_tensors = self._tokens_to_tensors(dev_tensors, max_length)
        test_tensors = self._tokens_to_tensors(test_tensors, max_length)

        return train_tensors, val_tensors, dev_tensors, test_tensors


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
    dataset = NewsGroupsDataset(feature_type, feature_config, model_type)
    dataset.load_data(
        train_data_filename,
        train_labels_filename,
        dev_data_filename,
        dev_labels_filename,
        test_data_filename,
    )

    # TODO: Featurize the input data for all three splits.
    print(f"> Featurizing the data...")
    return dataset.featurize()
