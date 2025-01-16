import re
import json


class BPETokenizer:
    
    def __init__(self, encoder_file="encoder.json", vocab_file="vocab.bpe"):
        with open(encoder_file, 'r', encoding='utf-8') as f:
            self.encoder = json.load(f)
        self.decoder = {v:k for k,v in self.encoder.items()}
        with open(vocab_file, 'r', encoding='utf-8') as f:
            vocab = f.read().split('\n')[1:-1]
        self.bpe_ranks = {tuple(line.split()): i for i, line in enumerate(vocab)}
        assert len(self.encoder) == 50257 and len(self.bpe_ranks) == 50000
        bs = list(range(33, 127)) + list(range(161, 256))
        xs = list(range(0, 33)) + list(range(127, 161))
        cs = bs[:] + [2**8 + i for i in range(len(xs))]
        self.byte_encoder = dict(zip(bs + xs, [chr(n) for n in cs]))
        self.byte_decoder = {v:k for k, v in self.byte_encoder.items()}

    def encode(self, text, allowed_special=None):
        tokens = re.findall(r"""<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d| ?""" +
                            r"""\w+| ?\d+| ?[^\s\w\d]+|\s+(?!\S)|\s+""", text, re.UNICODE)
        def translate(token):
            if token == '<|endoftext|>':
                assert allowed_special and token in allowed_special
                return [token]
            word = tuple(''.join(self.byte_encoder[byte] for byte in token.encode('utf-8')))
            while len(word) != 1:
                pairs = set((word[i], word[i+1]) for i in range(len(word)-1))
                bigram = min(pairs, key = lambda pair: self.bpe_ranks.get(pair, float('inf')))
                if bigram not in self.bpe_ranks:
                    break
                a, b = bigram
                new_word = []
                i = 0
                while i < len(word):
                    j = word.index(a, i) if a in word[i:] else len(word)
                    new_word.extend(word[i:j])
                    i = j
                    if i < len(word):
                        j = 2 if i < len(word)-1 and word[i] == a and word[i+1] == b else 1
                        new_word.append(a+b if j == 2 else word[i])
                        i += j
                word = tuple(new_word)
            return word
        return [self.encoder[_] for token in tokens for _ in translate(token)]

    def encode_batch(self, texts, allowed_special=None):
        """Encodes a batch of texts."""
        return [self.encode(text, allowed_special=allowed_special) for text in texts]

    def decode(self, tokens):
        tokens = [self.decoder[token] for token in tokens]
        buffer = bytearray([self.byte_decoder[c] for c in ''.join(tokens)])
        return buffer.decode('utf-8', errors='replace')

if __name__ == "__main__":

    PAD_TOKEN = 50256  # 应确保与 BPETokenizer 一致
    UNK_TOKEN = 1
    BOS_TOKEN = 50256
    EOS_TOKEN = 50256


    tokenizer = BPETokenizer(encoder_file="encoder.json", vocab_file="vocab.bpe")

    print("PAD_TOKEN ID:", tokenizer.encode("<pad>")[0] if "<pad>" in tokenizer.encoder else PAD_TOKEN)
    print("BOS_TOKEN ID:", tokenizer.encode("<bos>")[0] if "<bos>" in tokenizer.encoder else BOS_TOKEN)
    print("EOS_TOKEN ID:", tokenizer.encode("<eos>")[0] if "<eos>" in tokenizer.encoder else EOS_TOKEN)
    print("UNK_TOKEN ID:", tokenizer.encode("<unk>")[0] if "<unk>" in tokenizer.encoder else UNK_TOKEN)
