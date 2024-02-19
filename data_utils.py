import csv
import sys
import random


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
