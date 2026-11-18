
import torch

from Question_Generation.config.model_config import ModelConfig


class TextGenerator:

    def __init__(self, model, tokenizer):

        self.model = model
        self.tokenizer = tokenizer

        self.max_new_tokens = ModelConfig.MAX_NEW_TOKENS
        self.temperature = ModelConfig.TEMPERATURE
        self.max_length = model.max_length

    def _sample_next_token(
        self,
        logits,
        temperature,
        top_k
    ):

        if temperature <= 0:
            raise ValueError(
                "Temperature must be greater than zero."
            )

        logits = logits / temperature

        if top_k is not None:

            if top_k < 1:
                raise ValueError(
                    "top_k must be at least 1."
                )

            top_k = min(top_k, logits.size(-1))

            top_values, _ = torch.topk(
                logits,
                top_k
            )

            threshold = top_values[:, -1].unsqueeze(-1)

            logits = logits.masked_fill(
                logits < threshold,
                float("-inf")
            )

        probabilities = torch.softmax(
            logits,
            dim=-1
        )

        return torch.multinomial(
            probabilities,
            num_samples=1
        )

    def generate(
        self,
        prompt,
        max_new_tokens=None,
        temperature=None,
        top_k=10
    ):

        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens

        if temperature is None:
            temperature = self.temperature

        if max_new_tokens < 1:
            raise ValueError(
                "max_new_tokens must be greater than zero."
            )

        if temperature <= 0:
            raise ValueError(
                "Temperature must be greater than zero."
            )

        if top_k is not None and top_k < 1:
            raise ValueError(
                "top_k must be at least 1."
            )

        if not hasattr(self.tokenizer, "vocab"):
            raise ValueError(
                "Tokenizer vocabulary is not initialized."
            )

        if "[EOS]" not in self.tokenizer.vocab:
            raise ValueError(
                "Tokenizer must contain an [EOS] token."
            )

        input_ids = self.tokenizer.encode(prompt)

        if not input_ids:
            raise ValueError(
                "Prompt produced no tokens."
            )

        prompt_length = len(input_ids)

        if prompt_length > self.max_length:
            input_ids = input_ids[-self.max_length:]
            prompt_length = len(input_ids)

        try:
            model_device = next(
                self.model.parameters()
            ).device

        except (StopIteration, AttributeError):
            model_device = torch.device("cpu")

        input_tensor = torch.tensor(
            [input_ids],
            dtype=torch.long,
            device=model_device
        )

        eos_token_id = self.tokenizer.vocab["[EOS]"]

        self.model.eval()

        with torch.no_grad():

            for _ in range(max_new_tokens):

                model_input = input_tensor[:, -self.max_length:]

                logits = self.model(model_input)

                next_token_logits = logits[:, -1, :]

                next_token = self._sample_next_token(
                    next_token_logits,
                    temperature,
                    top_k
                )

                input_tensor = torch.cat(
                    [input_tensor, next_token],
                    dim=1
                )

                if next_token.item() == eos_token_id:
                    break

        generated_ids = input_tensor[
            0,
            prompt_length:
        ].tolist()

        return self.tokenizer.decode(generated_ids)
