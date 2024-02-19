import re


class TextProcessor:
    @staticmethod
    def basic_preprocess_text(text: str) -> str:
        """Preprocesses the input text."""
        # lowercase
        text = text.lower()
        # remove non-alphabetic characters
        text = re.sub(r"[^a-z]", " ", text)
        # remove extra spaces
        return re.sub(r"\s+", " ", text)

    @staticmethod
    def basic_stem_text(text: str) -> str:
        """Stems the input text."""

        def stem(word):
            if word.endswith("ing"):
                return word[:-3]
            if word.endswith("ed"):
                return word[:-2]
            return word

        # lowercase
        text = text.lower()
        # remove non-alphabetic characters
        text = re.sub(r"[^a-z]", " ", text)
        # remove extra spaces
        text = re.sub(r"\s+", " ", text)
        # stem the words
        return " ".join([stem(word) for word in text.split()])

    @staticmethod
    def enhanced_stem_text(text: str) -> str:
        """Stems the input text."""

        def enhanced_stemmer(word):
            # Convert to lowercase
            word = word.lower()

            # Handle some irregular forms with a simple lookup
            irregulars = {
                "went": "go",
                "mice": "mouse",
                "feet": "foot",
                "teeth": "tooth",
                "geese": "goose",
            }
            if word in irregulars:
                return irregulars[word]

            # Remove common plural forms and other endings
            if word.endswith("ies") and len(word) > 4:
                word = word[:-3] + "y"
            elif word.endswith("es") and len(word) > 3:
                # Avoid changing words like "goes" to "goe"
                if word[-3] not in "aeiou":
                    word = word[:-2]
            elif word.endswith("s") and len(word) > 3:
                word = word[:-1]
            # Convert past tense to present tense by removing "ed"
            if word.endswith("ed") and len(word) > 3:
                word = word[:-2]
            # Trim "ing" endings
            if word.endswith("ing") and len(word) > 4:
                word = word[:-3]
            # Remove "ly" endings
            if word.endswith("ly") and len(word) > 3:
                word = word[:-2]
            # Remove "er" or "est" endings (comparatives and superlatives)
            if word.endswith("er") and len(word) > 3:
                word = word[:-2]
            elif word.endswith("est") and len(word) > 4:
                word = word[:-3]
            # Trim "ization" endings, converting them to "ize"
            if word.endswith("ization") and len(word) > 7:
                word = word[:-5] + "e"

            return word

        # lowercase
        text = text.lower()
        # remove non-alphabetic characters
        text = re.sub(r"[^a-z]", " ", text)
        # remove extra spaces
        text = re.sub(r"\s+", " ", text)
        # stem the words
        return " ".join([enhanced_stemmer(word) for word in text.split()])
