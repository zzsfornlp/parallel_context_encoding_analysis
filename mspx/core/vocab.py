#

# Simple Vocab, flattened mapping between str <-> int

__all__ = [
    "VocabHelper", "Vocab"
]

from collections import OrderedDict, defaultdict
from typing import Iterable, Dict, List, Callable, IO, Type
import re
import os
import numpy as np
from mspx.utils import zlog, zopen, zwarn, Random, Registrable, Serializable, Configurable, Conf, default_json_serializer, default_pickle_serializer

# -----
class VocabHelper:
    # todo(note): these are reversed for other usage
    SPECIAL_PATTERN = re.compile(r"\<z_([a-zA-Z]+)_z\>")

    @staticmethod
    def extract_name(w):
        zmatch = re.fullmatch(VocabHelper.SPECIAL_PATTERN, w)
        if zmatch:
            return zmatch.group(1)
        else:
            return None

    @staticmethod
    def convert_special_pattern(w):
        return "<z_"+w+"_z>"

    # ----
    # find word with several backoff strategies
    # ordering of special normers
    WORD_NORMERS = [
        ["orig", lambda w: w],
        # ["num", lambda w: StrHelper.norm_num(w)],
        ["lc", lambda w: str.lower(w)],
        ["cased", lambda w: str.upper(w[0])+str.lower(w[1:])]
    ]

    # return (hit?, norm_name, normed_w)
    @staticmethod
    def norm_until_hit(v, w: str):
        # orig_w = w
        for norm_name, norm_f in VocabHelper.WORD_NORMERS:
            w = norm_f(w)
            if w in v:
                return True, norm_name, w
        return False, None, None

# =====
# the simple Str<->Integer vocab: no relations among the vocab entries
@Registrable.rd('V')
class Vocab(Serializable):
    def __init__(self, name="anon", **kwargs):
        self.name = name  # name for the vocab
        # todo(note): always keeps all these fields consistent: always keeping info of i2w, which has no repeats
        # fixed ones at the front and the end of the list
        self.pre_list = []
        self.post_list = []
        # these are real words
        self.w2i = {}  # word -> inner-idx (no specials)
        self.i2w = []  # inner-idx -> words
        self.counts = {}  # word -> counts
        # cache ones
        self._full_i2w = None  # outer-idx -> words
        # --
        # other confs & properties
        self.padding_side = 'right'
        self.model_max_length = 1000000000  # make it large!
        self.preset_is_char_vocab = False  # preset it as char-vocab
        self._is_char_vocab = None  # to do!
        if kwargs:
            self.__dict__.update(kwargs)
        # --

    @property
    def full_i2w(self):
        if self._full_i2w is None:
            self._full_i2w = [VocabHelper.convert_special_pattern(z) for z in self.pre_list] + self.i2w \
                             + [VocabHelper.convert_special_pattern(z) for z in self.post_list]
        return self._full_i2w

    @property
    def real_i2w(self):  # non-special real ones
        return self.i2w

    @property
    def idx_offset(self):
        return len(self.pre_list)

    def __len__(self):  # note: full length
        return len(self.i2w) + len(self.pre_list) + len(self.post_list)

    def __repr__(self):
        return f"Vocab[{self.name}]: len=({len(self.pre_list)}+{len(self.i2w)}+{len(self.post_list)})={len(self)}"

    def __contains__(self, item):
        return self.has_key(item)

    def __getitem__(self, item: str):
        assert self.has_key(item)
        return self.word2idx(item)

    def get(self, item, default=None):
        return self.word2idx(item, default)

    def has_key(self, item):
        return item in self.w2i

    # excluding pre and post ones
    def non_special_range(self):  # [)
        return (len(self.pre_list), len(self)-len(self.post_list))

    def non_speical_num(self):
        return len(self.i2w)

    def keys(self):
        return self.i2w

    # idx -> word
    def idx2word(self, idx: int):
        return self.full_i2w[idx]

    def word2idx(self, item, df=None):
        if item in self.w2i:
            return self.w2i[item] + self.idx_offset  # add offset to idx!!
        else:
            return df

    def seq_idx2word(self, idxes: List):
        return [self.full_i2w[ii] for ii in idxes]

    def seq_word2idx(self, words: List, df=None):
        return [self.word2idx(ww, df) for ww in words]

    # count related
    def word2count(self, item: str, df=0):
        if item in self.counts:
            return self.counts[item]
        else:
            return df

    def idx2count(self, idx: int):
        return self.counts[self.full_i2w[idx]]

    def get_all_counts(self):
        return sum(self.counts.values())

    # --
    # similar API to those in transformers.*Tokenizer
    def convert_tokens_to_ids(self, tokens): return self.seq_word2idx(tokens, df=self.unk)
    def convert_ids_to_tokens(self, ids): return self.seq_idx2word(ids)
    def get_vocab(self): return {vv: ii for ii, vv in enumerate(self.full_i2w)}

    @property
    def vocab_size(self): return len(self)
    @property
    def cls_token_id(self): return self.bos
    @property
    def sep_token_id(self): return self.eos
    @property
    def eos_token_id(self): return self.eos
    @property
    def pad_token_id(self): return self.pad
    @property
    def mask_token_id(self): return self.mask
    @property
    def unk_token_id(self): return self.unk
    @property
    def name_or_path(self): return self.name

    @property
    def is_char_vocab(self):
        if self._is_char_vocab is None:  # todo(+N): simple judgment!
            self._is_char_vocab = self.preset_is_char_vocab or (len(self.i2w)>0 and all(len(z)==1 for z in self.i2w))  # all items
        return self._is_char_vocab

    def tokenize(self, s: str):
        s2 = list(s) if self.is_char_vocab else s.split()  # simply splitting!
        return s2

    def decode(self, ids):
        toks = self.convert_ids_to_tokens(ids)
        c_join = "" if self.is_char_vocab else " "
        return c_join.join(toks)

    def __call__(self, text, add_special_tokens: bool = False, padding=False, return_tensors=None):
        # by default no adding special
        if isinstance(text, str):
            text = [text]
        toks = [self.tokenize(z) for z in text]  # bs*[L]
        masks = [[1]*len(z) for z in toks]
        if padding or return_tensors:
            pad_id = self.pad_token_id
            max_len = max([len(z) for z in toks])
            if self.padding_side == 'right':
                toks = [z+[pad_id]*(max_len-len(z)) for z in toks]
                masks = [z+[0]*(max_len-len(z)) for z in masks]
            else:
                toks = [[pad_id]*(max_len-len(z))+z for z in toks]
                masks = [[0]*(max_len-len(z))+z for z in masks]
        if return_tensors == 'pt':
            import torch
            toks = torch.as_tensor(toks, dtype=torch.int64)
            masks = torch.as_tensor(masks, dtype=torch.int64)
        else:
            assert return_tensors is None
        return {'input_ids': toks, 'attention_mask': masks}
    # --

    # =====
    # building related

    # DEFAULT_PRE_LIST, DEFAULT_POST_LIST = \
    #     tuple([VocabHelper.convert_special_pattern(w) for w in ["non"]]), \
    #     tuple([VocabHelper.convert_special_pattern(w)
    #            for w in ["unk", "eos", "bos", "pad", "mask"] + [f"spe{i}" for i in range(5)]])

    # note: make the special ones at the beginning!
    # DEFAULT_PRE_LIST, DEFAULT_POST_LIST = ("non", ), tuple(["unk", "eos", "bos", "pad", "mask"])
    DEFAULT_PRE_LIST, DEFAULT_POST_LIST = tuple(["non", "unk", "eos", "bos", "eod", "pad", "mask", "ex1", "ex2", "ex3"]), ()

    # basic settings
    def set_name(self, name: str):
        self.name = name

    def set_pre_post(self, pre_list: List = None, post_list: List = None):
        if pre_list is not None:
            self.pre_list = pre_list
        if post_list is not None:
            self.post_list = post_list
        self._build_props()
        self._full_i2w = None  # clear!

    def set_i2w(self, new_i2w: List[str]):
        self._rebuild_items(new_i2w)

    # build public properties for the special tokens
    def _build_props(self):
        for ii, name in enumerate(self.pre_list):
            assert not hasattr(self, name)
            setattr(self, name, ii)
        for ii, name in enumerate(self.post_list):
            assert not hasattr(self, name)
            setattr(self, name, ii+len(self.pre_list)+len(self.i2w))

    # =====
    # feeding
    def feed_one(self, w: str, c=1):
        counts = self.counts
        is_new_entry = False
        if w not in counts:
            # also keep them in the adding order
            self.i2w.append(w)
            self.w2i[w] = len(self.w2i)
            counts[w] = c
            is_new_entry = True
            # --
            self.set_pre_post()  # remember to refresh special ones!!
            # --
        else:
            counts[w] += c
        return is_new_entry

    def feed_iter(self, iter: Iterable):
        rets = []  # whether add new entry
        for w in iter:
            rets.append(self.feed_one(w))
        return rets

    # filtering and sort
    def _rebuild_items(self, new_i2w: List[str], default_count=0, by_what=""):
        # todo(note): change inside!
        before_str = str(self)
        self.i2w = new_i2w
        self.w2i = {k:i for i,k in enumerate(new_i2w)}
        assert len(self.i2w) == len(self.w2i), "Err: repeated items in new_i2w!!"
        old_counts = self.counts
        self.counts = {k:old_counts.get(k, default_count) for k in new_i2w}
        after_str = str(self)
        # --
        self.set_pre_post()  # remember to refresh special ones!!
        # --
        zlog(f"Rebuild Vocab by {by_what}: {before_str} -> {after_str}")

    # filter out certain items
    def build_filter(self, word_filter=(lambda w, i, c: True)):
        _counts = self.counts
        new_i2w = [w for i,w in enumerate(self.i2w) if word_filter(w,i,_counts[w])]
        self._rebuild_items(new_i2w, by_what="filter")

    # shortcut
    def build_filter_thresh(self, rthres: int, fthres: int):
        self.build_sort()  # must sort to allow rank-thresh
        def rf_filter(w, i, c): return c>=fthres and i<=rthres
        self.build_filter(rf_filter)

    # sort items: by default sort by -count, adding-idx, word-str
    def build_sort(self, key=lambda w, i, c: (-c, w)):
        _counts = self.counts
        sorting_info = [(key(w,i,_counts[w]), w) for i,w in enumerate(self.i2w)]  # put key at first
        assert len(sorting_info) == len(self.i2w), "Inner error, repeated key!"
        sorting_info.sort()
        new_i2w = [z[1] for z in sorting_info]
        self._rebuild_items(new_i2w, by_what="sort")

    # add new tokens
    def add_tokens(self, tokens):
        ret = 0
        for t in tokens:
            ret += int(self.feed_one(t))
        zlog(f"Try to add extra_tokens {tokens}: {ret} added!")
        return ret

    # -----
    # return a pandas table
    def get_info_table(self):
        d = Vocab.create_info_table(self.i2w, [self.word2count(w) for w in self.i2w], [self.get(w) for w in self.i2w])
        return d

    @staticmethod
    def create_info_table(words, counts, idxes=None):
        import pandas as pd
        if idxes is None:
            idxes = list(range(len(words)))
        # --
        res = []
        accu_counts = 0
        for ii, w in enumerate(words):
            i, c = idxes[ii], counts[ii]
            accu_counts += c
            res.append([i, w, c, 0., accu_counts, 0.])
        d = pd.DataFrame(res, columns=["Idx", "Word", "Count", "Perc.", "ACount", "APerc."])
        d["Perc."] = d["Count"] / accu_counts
        d["APerc."] = d["ACount"] / accu_counts
        return d

    # =====
    # some shortcut buildings

    # build for toker
    @staticmethod
    def build_empty_toker(name="toker", **kwargs):
        return Vocab(name=name, **kwargs)

    # build empty
    @staticmethod
    def build_empty(name="anon", pre_list=DEFAULT_PRE_LIST, post_list=DEFAULT_POST_LIST, **kwargs):
        v = Vocab(name=name, **kwargs)
        v.set_pre_post(pre_list, post_list)
        return v

    # build with static items
    @staticmethod
    def build_by_static(items: List[str], name="anon", pre_list=None, post_list=None, **kwargs):
        v = Vocab(name=name, **kwargs)
        v.set_i2w(items)
        v.set_pre_post(pre_list, post_list)
        return v

    # build from counting iters
    # word_filter=(lambda w, i, c: True)
    @staticmethod
    def build_from_iter(iter: Iterable, name="anon", pre_list=DEFAULT_PRE_LIST, post_list=DEFAULT_POST_LIST,
                        sorting=False, word_filter: Callable = None, **kwargs):
        v = Vocab(name=name, **kwargs)
        for one in iter:
            v.feed_one(one)
        v.set_pre_post(pre_list, post_list)
        if sorting:  # first do possible sorting!!
            v.build_sort()
        if word_filter is not None:
            v.build_filter(word_filter)
        return v

    # merge multiple vocabs
    @staticmethod
    def merge_vocabs(vocabs: Iterable['Vocab'], name="merged", pre_list=DEFAULT_PRE_LIST, post_list=DEFAULT_POST_LIST,
                     sorting=False, word_filter: Callable = None):
        v = Vocab(name=name)
        for one_vocab in vocabs:
            for w in one_vocab.keys():  # note: add counts
                v.feed_one(w, one_vocab.word2count(w))
        v.set_pre_post(pre_list, post_list)
        if sorting:  # first do possible sorting!!
            v.build_sort()
        if word_filter is not None:
            v.build_filter(word_filter)
        return v

    # filter inits for embeddings
    def filter_embed(self, wv: 'WordVectors', init_nohit=None, scale=1.0, assert_all_hit=False):
        if init_nohit is None:  # auto decide by wv
            init_nohit = np.mean([np.std(z) for z in wv.vecs]).item()
            zlog(f"Auto decide init_nohit={init_nohit}")
        if init_nohit <= 0.:
            get_nohit = lambda s: np.zeros((s,), dtype=np.float32)
        else:
            _generator = Random.get_generator("vocab")
            get_nohit = lambda s: _generator.standard_normal(s) * init_nohit
        # --
        ret = []
        res = defaultdict(int)
        embed_size = wv.get_emb_size()
        # for w in self.keys():  # todo(+N): once a bug!
        for w in self.full_i2w:
            hit, norm_name, norm_w = wv.norm_until_hit(w)
            if hit:
                value = np.asarray(wv.get_vec(norm_w, norm_name=False), dtype=np.float32)
                res[norm_name] += 1
            else:
                value = get_nohit(embed_size)
                # value = np.zeros((embed_size,), dtype=np.float32)
                res["no-hit"] += 1
            ret.append(value)
        # --
        if assert_all_hit:
            assert res["no-hit"]==0, f"Filter-embed error: assert all-hit but get no-hit of {res['no-hit']}"
        zret = np.asarray(ret, dtype=np.float32) * scale
        zlog(f"Filter pre-trained embed {self}->{zret.shape}: {res}, no-hit is inited with {init_nohit}.")
        return zret

    # ======
    # special method for combining embeddings and vocabs
    # keep main's pre&post, but put aug's true words before post and make corresponding changes to the arr
    # -> return (new_vocab, new_arr)
    @staticmethod
    def aug_vocab_and_arr(main_vocab: 'Vocab', main_arr, aug_vocab: 'Vocab', aug_arr, new_name='aug'):
        # first merge the vocab
        new_vocab = Vocab.merge_vocabs(
            [main_vocab, aug_vocab], name=new_name, sorting=False, pre_list=main_vocab.pre_list, post_list=main_vocab.post_list)
        # then find the arrays
        new_arr = [main_arr[i] for i in range(len(main_vocab.pre_list))]
        main_hit = aug_hit = 0
        for idx in range(*(new_vocab.non_special_range())):
            word = new_vocab.idx2word(idx)
            # note: selecting the embeds in aug first (make it possible to override original ones!)
            aug_orig_idx = aug_vocab.get(word)
            if aug_orig_idx is None:
                main_orig_idx = main_vocab[word]  # must be there!
                new_arr.append(main_arr[main_orig_idx])
                main_hit += 1
            else:
                new_arr.append(aug_arr[aug_orig_idx])
                aug_hit += 1
        new_arr.extend([main_arr[i] for i in range(-len(main_vocab.post_list), 0)])
        # --
        zlog(f"For the final merged arr, the composition is all={len(new_arr)},main={main_hit},aug={aug_hit}")
        ret_arr = np.asarray(new_arr)
        return new_vocab, ret_arr

    # --
    # shortcuts for save & load
    def save(self, filename: str):
        return self.to_file(filename)

    @classmethod
    def load(cls, filename: str):
        return cls.create_from_file(filename)
