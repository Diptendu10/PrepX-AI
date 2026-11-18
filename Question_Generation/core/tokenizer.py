
import re


class Tokenizer:

    def __init__(self):

        # Special tokens
        self.special_tokens = [
            "[PAD]",
            "[UNK]",
            "[BOS]",
            "[EOS]"
        ]

        self.vocab = {}

        self.id_to_token = {}

    # 1. Divide sentence into tokens
    def tokenize(self, text):

        text = text.lower()

        tokens = re.findall(
            r"\w+|[^\w\s]",
            text
        )

        return tokens

    # 2. Create vocabulary
    def build_vocab(self, texts):

        vocabulary = set()

        for text in texts:

            tokens = self.tokenize(text)

            vocabulary.update(tokens)

        all_tokens = self.special_tokens + sorted(vocabulary)

        self.vocab = {
            token: index
            for index, token in enumerate(all_tokens)
        }

        self.id_to_token = {
            index: token
            for token, index in self.vocab.items()
        }

    # 3. Convert tokens into numerical IDs
    def encode(self, text):

        tokens = self.tokenize(text)

        token_ids = [
            self.vocab.get(token, self.vocab["[UNK]"])
            for token in tokens
        ]

        return (
            [self.vocab["[BOS]"]]
            + token_ids
            + [self.vocab["[EOS]"]]
        )

    # 4. Convert numerical IDs back into text
    def decode(self, token_ids):

        tokens = [
            self.id_to_token.get(token_id, "[UNK]")
            for token_id in token_ids
        ]

        tokens = [
            token for token in tokens
            if token not in self.special_tokens
        ]

        return " ".join(tokens)

    # 5. Display vocabulary
    def get_vocab(self):

        return self.vocab
