import math
import timm
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader, TensorDataset
import loralib as lora
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from p2_legacy.p2_decoder_lora import DecoderLoRA

PAD_TOKEN = 50256
UNK_TOKEN = 1
BOS_TOKEN = 50256
EOS_TOKEN = 50256

class ImageCaptioningTransformer(nn.Module):

    PAD_TOKEN = 50256
    UNK_TOKEN = 1
    BOS_TOKEN = 50256
    EOS_TOKEN = 50256

    def __init__(self, vocab_size, encoder, num_layers, nhead, d_model, activation='gelu', batch_first=True, dropout=0.1, pretrained=True) -> None:
        super().__init__()
        self.config = dict(locals())
        del self.config['self']
        for k in dict(locals()):
            if k.startswith('_'):
                del self.config[k]

        self.word_embedding = nn.Embedding(
            vocab_size, d_model, padding_idx=self.PAD_TOKEN)
        self.positional_embedding = PositionalEmbedding(d_model=d_model)

        self.encoder = timm.create_model(
            encoder, pretrained=pretrained, num_classes=0)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            activation=activation,
            batch_first=batch_first,
            dropout=dropout,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=num_layers,
        )
        self.head = nn.Linear(d_model, vocab_size)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.PAD_TOKEN)

    def forward(self, batch_image, input_ids, sampleing_ratio=0):
        # encoder pass
        features = self.encoder.forward_features(
            batch_image)  # shape = (B, 14*14+1, d_model)

        if sampleing_ratio > 0:
            with torch.no_grad():
                in_embed = self.word_embedding(input_ids[:, :-1])
                in_embed += self.positional_embedding(in_embed)
                mask = self._generate_square_subsequent_mask(
                    in_embed.shape[1]).to(in_embed.device)
                logits = self.decoder(
                    tgt=in_embed, memory=features, tgt_mask=mask)
                logits = self.head(logits)  # shape (B, seq_len, vocab)
            pred_ids = logits.argmax(dim=-1)  # (B, seq_len)
            # place BOS in pred
            pred_ids = torch.cat(
                [self.BOS_TOKEN * torch.ones(pred_ids.shape[0], 1, dtype=input_ids.dtype).to(input_ids.device), pred_ids], dim=1)

            # replace input_ids
            '''
            loop-style:
            for batch_idx in range(input_ids.shape[0]):
                for seq_idx in range(1, input_ids.shape[1]):
                    if input_ids[batch_idx][seq_idx] == self.EOS_TOKEN:  # EOS
                        break
                    if random.random() > sampleing_ratio:
                        # pred_ids[B][0] = output of 0th word -> replaces the 1st word
                        input_ids[batch_idx][seq_idx] = pred_ids[batch_idx][seq_idx]
            '''
            to_be_replaced = torch.rand_like(
                pred_ids, dtype=float) > sampleing_ratio
            # don't replace PAD and EOS
            to_be_replaced = torch.logical_and(
                to_be_replaced,
                input_ids != self.PAD_TOKEN
            )
            to_be_replaced = torch.logical_and(
                to_be_replaced,
                input_ids != self.EOS_TOKEN
            )
            input_ids[to_be_replaced] = pred_ids[to_be_replaced]

        # word embedding + positional embedding
        # 0 ~ n-1 TOKENs as input
        in_embed = self.word_embedding(input_ids[:, :-1])
        in_embed += self.positional_embedding(in_embed)

        # decoder pass
        mask = self._generate_square_subsequent_mask(
            in_embed.shape[1]).to(in_embed.device)
        logits = self.decoder(tgt=in_embed, memory=features, tgt_mask=mask)
        logits = self.head(logits)  # shape (B, seq_len, vocab)

        # 1 ~ n TOKENs as target output
        logits = torch.swapaxes(logits, 1, 2)
        loss = self.criterion(logits, input_ids[:, 1:])
        return loss

    # https://github.com/pytorch/examples/blob/5551061414d3bcf202de520d20e8163f58eb664a/word_language_model/model.py#L126
    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float(
        ).masked_fill(
            mask == 0, float('-inf')
        ).masked_fill(
            mask == 1, float(0.0)
        )
        return mask

    def greedy_search(self, img, max_length=30):
        self.eval()
        if img.dim() < 4:
            img = img.unsqueeze(0)
        device = img.device
        with torch.no_grad():
            memory = self.encoder.forward_features(img)

        current_state = torch.tensor([self.BOS_TOKEN]).to(device).unsqueeze(1)
        for _ in range(max_length):
            with torch.no_grad():
                in_embed = self.word_embedding(current_state)
                in_embed += self.positional_embedding(in_embed)
                logits = self.decoder(tgt=in_embed, memory=memory)
                logits = self.head(logits[:, -1, :])
            next_word = logits.argmax(dim=-1).unsqueeze(0)
            if next_word.item() == self.EOS_TOKEN:
                break
            current_state = torch.concat((current_state, next_word), dim=-1)
        return current_state[0, 1:].cpu().tolist()  # remove [BOS]

    @torch.no_grad()
    def batch_beam_search(self, img, beams=3, max_length=30):
        def forward_prob(memory, input_ids):
            in_embed = self.word_embedding(input_ids)
            in_embed += self.positional_embedding(in_embed)
            logits = self.decoder(tgt=in_embed, memory=memory)
            logits = self.head(logits[:, -1, :])
            return logits

        self.eval()
        if img.dim() < 4:
            img = img.unsqueeze(0)
        memory = self.encoder.forward_features(img)

        device = memory.device
        current_state = self.BOS_TOKEN * \
            torch.ones(memory.shape[0], 1, dtype=torch.long).to(device)

        next_probs = forward_prob(memory, current_state)
        vocab_size = next_probs.shape[-1]
        probs, next_chars = \
            next_probs.reshape(next_probs.shape[0], next_probs.shape[-1])\
            .log_softmax(-1)\
            .topk(k=beams, axis=-1)  # shape = (B, topk)
        current_state = current_state.repeat((beams, 1))
        next_chars = next_chars.reshape(-1, 1)
        current_state = torch.cat((current_state, next_chars), axis=1)
        memory = memory.repeat((beams, 1, 1))

        for i in range(2, max_length + 1):
            next_probs = forward_prob(memory, current_state).log_softmax(-1)
            next_probs = next_probs.reshape((-1, beams, next_probs.shape[-1]))
            probs = probs.unsqueeze(-1) + next_probs
            probs = probs.flatten(start_dim=1)
            probs, idx = probs.topk(k=beams, axis=-1)
            next_chars = torch.remainder(
                idx, vocab_size).flatten().unsqueeze(-1)
            top_candidates = (idx / vocab_size).long()
            top_candidates += torch.arange(current_state.shape[0] //
                                           beams, device=device).unsqueeze(-1) * beams
            current_state = current_state[top_candidates].flatten(end_dim=-2)
            current_state = torch.cat([current_state, next_chars], axis=1)

        current_state = current_state.reshape(-1,
                                              beams, current_state.shape[-1])
        current_state = current_state[..., 1:]  # remove BOS
        # get best Y
        best_idx = probs.argmax(-1)
        current_state = current_state[torch.arange(len(best_idx)), best_idx, :]
        exit()
        return current_state.cpu().tolist()

    # reference: https://github.com/jarobyte91/pytorch_beam_search
    @torch.no_grad()
    def beam_search(self, img, beams=3, max_length=30):
        def forward_prob(memory, input_ids):
            in_embed = self.word_embedding(input_ids)
            in_embed += self.positional_embedding(in_embed)
            logits = self.decoder(tgt=in_embed, memory=memory)
            logits = self.head(logits[:, -1, :])
            return logits

        self.eval()
        if img.dim() < 4:
            img = img.unsqueeze(0)
        memory = self.encoder.forward_features(img)
        device = memory.device
        current_state = torch.tensor([self.BOS_TOKEN]).to(device).unsqueeze(1)

        # get top k words
        next_probs = forward_prob(memory, current_state)
        vocab_size = next_probs.shape[-1]
        current_probs, next_chars = next_probs.log_softmax(
            -1).topk(k=beams, axis=-1)
        current_probs = current_probs.reshape(beams)
        next_chars = next_chars.reshape(beams, 1)

        # gen first k beams
        current_state = current_state.repeat((beams, 1))
        current_state = torch.cat((current_state, next_chars), axis=1)

        ans_ids = []
        ans_probs = []
        for i in range(max_length - 1):
            # get top k beams for beam*beam candidates
            next_probs = forward_prob(memory.repeat(
                (beams, 1, 1)), current_state).log_softmax(-1)
            current_probs = current_probs.unsqueeze(-1) + next_probs
            current_probs = current_probs.flatten()  # (beams*vocab)

            # length normalization
            _, idx = (current_probs / (len(current_state[0]) + 1))\
                .topk(k=beams, dim=-1)
            current_probs = current_probs[idx]

            # get corresponding next char
            next_chars = torch.remainder(idx, vocab_size)
            next_chars = next_chars.unsqueeze(-1)

            # get corresponding original beams
            top_candidates = (idx / vocab_size).long()
            current_state = current_state[top_candidates]
            current_state = torch.cat((current_state, next_chars), dim=1)

            # concat next_char to beams
            to_rm_idx = set()
            for idx, ch in enumerate(next_chars):
                if i == (max_length - 2) or ch.item() == self.EOS_TOKEN:
                    ans_ids.append(current_state[idx].cpu().tolist())
                    ans_probs.append(
                        current_probs[idx].item() / len(ans_ids[-1]))
                    to_rm_idx.add(idx)
                    beams -= 1
            to_keep_idx = [i for i in range(
                len(current_state)) if i not in to_rm_idx]
            if len(to_keep_idx) == 0:
                break
            current_state = current_state[to_keep_idx]
            current_probs = current_probs[to_keep_idx]

        max_idx = torch.argmax(torch.tensor(ans_probs)).item()
        return ans_ids[max_idx]


class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, max_len=512):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() *
                    -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


# https://github.com/ildoonet/pytorch-gradual-warmup-lr/blob/master/warmup_scheduler/scheduler.py
class GradualWarmupScheduler(_LRScheduler):
    
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError(
                'multiplier should be greater thant or equal to 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [
                        base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        self.last_epoch = epoch if epoch != 0 else 1
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch /
                                    self.total_epoch + 1.) for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
                self._last_lr = self.after_scheduler.get_last_lr()
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)


class Config:
    def __init__(self, checkpoint=None, rank=16):
        self.n_layer = 12
        self.n_head = 12
        self.n_embd = 768
        self.vocab_size = 50257
        self.block_size = 1024
        self.checkpoint = checkpoint
        self.rank = rank  # Rank for LoRA


class ImageCaptioningTransformerLoRA(nn.Module):

    PAD_TOKEN = 50256
    UNK_TOKEN = 1
    BOS_TOKEN = 50256
    EOS_TOKEN = 50256

    def __init__(self, vocab_size, encoder, num_layers, nhead, d_model, rank=4, activation='gelu', batch_first=True, dropout=0.1, pretrained=True, decoder_path="/home/nas/polin/DLCV/dlcv-fall-2024-hw3-berlin0308/hw3_data/p2_data/decoder_model.bin") -> None:
        super().__init__()
        self.config = dict(locals())
        del self.config['self']
        for k in dict(locals()):
            if k.startswith('_'):
                del self.config[k]

        # Initialize embedding with LoRA on the word_embedding layer
        self.word_embedding = lora.Embedding(vocab_size, d_model, padding_idx=self.PAD_TOKEN, r=rank)
        self.positional_embedding = PositionalEmbedding(d_model=d_model)

        # Load pretrained encoder with LoRA applied
        self.encoder = timm.create_model(
            encoder, pretrained=pretrained, num_classes=0
        )
        self.feature_resize = nn.Linear(1024, 768)

        self.cfg = Config(decoder_path)
        self.decoder = DecoderLoRA(self.cfg)

        # Output projection layer with LoRA applied
        self.head = lora.Linear(d_model, vocab_size, r=rank)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.PAD_TOKEN)

    def forward(self, batch_image, input_ids, sampleing_ratio=0):

        # print(batch_image.shape)
        # print(input_ids.shape)
        # print(input_ids)

        # Encoder pass
        features = self.encoder.forward_features(batch_image)  # shape = (B, 14*14+1, d_model)
        features = self.feature_resize(features)

        logits = self.decoder(x=input_ids, encoder_feature=features) #, tgt_mask=mask)
        
        # Crop logits to match input_ids' sequence length
        B, T_in, D = input_ids.shape[0], input_ids.shape[1], logits.shape[-1]
        logits = logits[:, :T_in, :]  # shape now (B, T_in, D), where D == vocab_size (50257)
        # print("Adjusted logits shape:", logits.shape)

        # Compute loss
        logits = torch.swapaxes(logits, 1, 2)  # Change to (B, vocab_size, T_in) for CrossEntropyLoss
        loss = self.criterion(logits, input_ids)
        return loss


    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def greedy_search(self, img, max_length=30):
        self.eval()
        if img.dim() < 4:
            img = img.unsqueeze(0)
        device = img.device
        with torch.no_grad():
            memory = self.encoder.forward_features(img)

        current_state = torch.tensor([self.BOS_TOKEN]).to(device).unsqueeze(1)
        for _ in range(max_length):
            with torch.no_grad():
                in_embed = self.word_embedding(current_state)
                in_embed += self.positional_embedding(in_embed)
                logits = self.decoder(tgt=in_embed, memory=memory)
                logits = self.head(logits[:, -1, :])
            next_word = logits.argmax(dim=-1).unsqueeze(0)
            if next_word.item() == self.EOS_TOKEN:
                break
            current_state = torch.cat((current_state, next_word), dim=-1)
        return current_state[0, 1:].cpu().tolist()
    
    def beam_search(self, img, beams=3, max_length=30, device="cuda"):
        self.eval()

        def forward_prob(x: Tensor, encoder_feature: Tensor):
            
            x = torch.narrow(x, 1, 0, min(x.size(1), self.decoder.block_size))
            pos = torch.arange(x.size()[1], dtype=torch.long, device=x.device).unsqueeze(0)

            x = self.decoder.transformer.wte(x) + self.decoder.transformer.wpe(pos)
            x = torch.cat((encoder_feature, x), dim=1)

            for block in self.decoder.transformer.h:
                x = block(x)

            x = self.decoder.lm_head(self.decoder.transformer.ln_f(x[:, -1, :]))
            return x

        if img.dim() < 4:
            img = img.unsqueeze(0)

        encoder_feature = self.encoder.forward_features(img)
        encoder_feature = self.feature_resize(encoder_feature)
        cur_state = torch.tensor([BOS_TOKEN]).to(device).unsqueeze(1)
        ### Beam Search Start ###
        # get top k words
        next_probs = forward_prob(cur_state, encoder_feature)

        vocab_size = next_probs.shape[-1]

        # probs, pred id
        cur_probs, next_chars = next_probs.log_softmax(-1).topk(k=beams, axis=-1)
        cur_probs = cur_probs.reshape(beams)
        next_chars = next_chars.reshape(beams, 1)
        # gen first k beams
        cur_state = cur_state.repeat((beams, 1))  # 複製 beams 次
        cur_state = torch.cat((cur_state, next_chars), axis=1)

        ans_ids = []
        ans_probs = []
        for i in range(max_length - 1):
            # get top k beams for beam*beam candidates
            # print("current state: ", cur_state)
            next_probs = forward_prob(
                cur_state, encoder_feature.repeat((beams, 1, 1))
            ).log_softmax(-1)
            cur_probs = cur_probs.unsqueeze(-1) + next_probs
            cur_probs = cur_probs.flatten()  # (beams*vocab) 攤平成1D

            # length normalization
            # cur_probs / (len(cur_state[0]) + 1) -> nomalized
            _, idx = (cur_probs / (len(cur_state[0]) + 1)).topk(k=beams, dim=-1)
            cur_probs = cur_probs[idx]

            # get corresponding next char
            next_chars = torch.remainder(idx, vocab_size)
            next_chars = next_chars.unsqueeze(-1)
            # print("next char: ",next_chars)

            # get corresponding original beams
            top_candidates = (idx / vocab_size).long()  # 找回屬於哪個beam
            cur_state = cur_state[top_candidates]
            cur_state = torch.cat((cur_state, next_chars), dim=1)

            # concat next_char to beams
            to_rm_idx = set()
            for idx, ch in enumerate(next_chars):
                if i == (max_length - 2) or ch.item() == EOS_TOKEN:
                    ans_ids.append(cur_state[idx].cpu().tolist())
                    # print(cur_probs[idx].item()," / ",len(ans_ids[-1]))
                    ans_probs.append(cur_probs[idx].item() / len(ans_ids[-1]))
                    to_rm_idx.add(idx)
                    beams -= 1

            to_keep_idx = [i for i in range(len(cur_state)) if i not in to_rm_idx]
            if len(to_keep_idx) == 0:
                break
            cur_state = cur_state[to_keep_idx]
            cur_probs = cur_probs[to_keep_idx]

        max_idx = torch.argmax(torch.tensor(ans_probs)).item()

        # 把50256抽離
        ans_ids[max_idx] = [x for x in ans_ids[max_idx] if x != EOS_TOKEN]
        # print(ans_ids)
        return ans_ids[max_idx]

