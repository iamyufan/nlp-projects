from typing import List, Dict, Any

from text_preprocessor import TextProcessor


class Vocabulary:
    def __init__(self, feature_config: Dict[str, Any]):
        self.itos = {0: "<PAD>", 1: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<UNK>": 1}
        
        self.feature_config = feature_config

        # Set the N for N-grams
        self.n = feature_config.get("n", 1)

        # Load the stopwords
        self.remove_stopwords = feature_config.get("remove_stopwords", True)
        if self.remove_stopwords:
            with open("stopwords.txt", "r") as file:
                stopwords = file.read().splitlines()
            self.stopwords = stopwords

        # Set the text processor
        self.text_processor = TextProcessor.basic_preprocess_text

        # Set the size of the vocabulary
        # 1. Set it with the max_vocab_size if provided
        self.max_vocab_size = feature_config.get("max_vocab_size", float("inf"))
        # 2. Set it with the min_count if provided
        self.min_count = feature_config.get("min_count", 1)

        # Initialize the IDF dictionary
        self.weighting = feature_config.get("weighting", "binary")
        if self.weighting == "tfidf":
            self.idf = {}

    def __len__(self):
        if len(self.itos) != len(self.stoi):
            raise ValueError("Length mismatch between itos and stoi")
        return len(self.itos)

    def tokenize(self, sentence: str):
        """Tokenizes a sentence."""
        sentence = self.text_processor(sentence)
        tokens = sentence.split()

        # Remove stopwords
        if self.remove_stopwords:
            tokens = [word for word in tokens if word not in self.stopwords]

        # Generate n-grams if n > 1
        if self.n > 1:
            tokens = [
                " ".join(tokens[i : i + self.n])
                for i in range(len(tokens) - self.n + 1)
            ]
        
        return tokens

    def build_vocabulary_from_data(self, data: List[str]):
        """Builds a vocabulary from a list of raw sentences."""
        tokenized_data = [self.tokenize(sentence) for sentence in data]

        # 1. Filter out the tokens with the number of sentences
        # they appear in is less than min_count
        token_counts = {}
        for tokens in tokenized_data:
            for token in tokens:
                token_counts[token] = token_counts.get(token, 0) + 1
        tokenized_data = [
            [token for token in tokens if token_counts[token] >= self.min_count]
            for tokens in tokenized_data
        ]

        # 2. Build the vocabulary with the top max_vocab_size tokens
        token_counts = {}
        for tokens in tokenized_data:
            for token in tokens:
                token_counts[token] = token_counts.get(token, 0) + 1
        token_counts = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)

        for i, (token, count) in enumerate(token_counts):
            if i + 2 >= self.max_vocab_size:
                break
            self.itos[i + 2] = token
            self.stoi[token] = i + 2

        # 3. Compute the IDF dictionary if the weighting is tfidf
        if self.weighting == "tfidf":
            N = len(tokenized_data)
            for token in self.stoi:
                df = sum([1 for tokens in tokenized_data if token in tokens])
                self.idf[token] = N / df

    def encode(self, sentence: str):
        """Encodes a sentence."""
        tokens = self.tokenize(sentence)
        return [self.stoi.get(token, self.stoi["<UNK>"]) for token in tokens]
