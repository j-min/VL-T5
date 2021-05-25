from transformers import T5Tokenizer, PreTrainedTokenizer
import re
import sentencepiece as spm

class VLT5Tokenizer(T5Tokenizer):

    # vocab_files_names = VOCAB_FILES_NAMES
    # pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    # max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    # model_input_names = ["attention_mask"]

    def __init__(
        self,
        vocab_file,
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        extra_ids=100,
        vis_extra_ids=100,
        additional_special_tokens=None,
        **kwargs
    ):
        # Add extra_ids to the special token list
        if extra_ids > 0 and additional_special_tokens is None:
            additional_special_tokens = [
                "<extra_id_{}>".format(i) for i in range(extra_ids)]

        elif extra_ids > 0 and additional_special_tokens is not None:
            # Check that we have the right number of extra_id special tokens
            extra_tokens = len(
                set(filter(lambda x: bool("extra_id" in x), additional_special_tokens)))
            if extra_tokens != extra_ids:
                raise ValueError(
                    f"Both extra_ids ({extra_ids}) and additional_special_tokens ({additional_special_tokens}) are provided to T5Tokenizer. "
                    "In this case the additional_special_tokens must include the extra_ids tokens"
                )

        if vis_extra_ids > 0:
            additional_special_tokens.extend(["<vis_extra_id_{}>".format(i) for i in range(vis_extra_ids)])

        PreTrainedTokenizer.__init__(
            self,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            extra_ids=extra_ids,
            additional_special_tokens=additional_special_tokens,
            **kwargs,
        )

        self.vocab_file = vocab_file
        self._extra_ids = extra_ids
        self._vis_extra_ids = vis_extra_ids

        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(vocab_file)

    @property
    def vocab_size(self):
        return self.sp_model.get_piece_size() + self._extra_ids + self._vis_extra_ids

    def get_vocab(self):
        vocab = {self.convert_ids_to_tokens(
            i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def _convert_token_to_id(self, token):
        """ Converts a token (str) in an id using the vocab. """
        if token.startswith("<extra_id_"):
            match = re.match(r"<extra_id_(\d+)>", token)
            num = int(match.group(1))
            return self.vocab_size - num - 1 - self._vis_extra_ids
        elif token.startswith("<vis_extra_id_"):
            match = re.match(r"<vis_extra_id_(\d+)>", token)
            num = int(match.group(1))
            return self.vocab_size - num - 1
        return self.sp_model.piece_to_id(token)

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        if index < self.sp_model.get_piece_size():
            token = self.sp_model.IdToPiece(index)
        else:
            if index > self.sp_model.get_piece_size() + self._extra_ids - 1:
                token = "<vis_extra_id_{}>".format(self.vocab_size - 1 - index)
            else:
                token = "<extra_id_{}>".format(self.vocab_size - self._vis_extra_ids - 1 - index)
        return token