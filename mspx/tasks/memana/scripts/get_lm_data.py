#

# to obtain testing data for LM PPL testing

import sys
import json
from collections import Counter
from datasets import load_dataset
from mspx.utils import get_global_cache_dir, zlog, default_json_serializer

# --
# some utils from mspx.tasks.mygo.scripts.data.prep_data
def do_load_dataset(path: str, **kwargs):
    cache_dir = get_global_cache_dir()
    splits = path.split(":")
    if len(splits) < 3:
        splits = splits + [None, None]
    _path, _name, _split = splits[:3]
    _split = {"dev": "validation"}.get(_split, _split)
    ret = load_dataset(_path, _name, split=_split, cache_dir=cache_dir, **kwargs)
    zlog(f"Load dataset {splits} -> {ret}")
    return ret

def yield_pg19(split_names=("dev", )):
    for split_name in split_names:
        _path = f"deepmind/pg19::{split_name}"
        ds = do_load_dataset(_path)
        for inst in ds:
            new_inst = {"info": {k: inst[k] for k in ["short_book_title", "publication_date", "url"]}, "text": inst["text"]}
            yield new_inst

def yield_proof_pile(split_names=("dev", )):
    for split_name in split_names:
        _path = f"hoskinson-center/proof-pile::{split_name}"
        ds = do_load_dataset(_path)
        for inst in ds:
            meta = json.loads(inst['meta'])
            if meta.get('config', None) == "arxiv":  # filter arxiv ones
                new_inst = {"info": meta, "text": inst["text"]}
                yield new_inst
    # --

# --
def main(dataname: str, output_file: str):
    # first get all data
    ff = globals()[f"yield_{dataname}"]
    cc = Counter()
    all_insts = []
    for inst in ff():
        cc['inst'] += 1
        cc['token'] += len(inst["text"].split())
        cc['char'] += len(inst["text"])
        all_insts.append(inst)
    zlog(f"Get data for {dataname}: {cc}")
    default_json_serializer.save_iter(all_insts, output_file)
    # --

# --
# python -mpdb -m mspx.tasks.memana.scripts.get_lm_data
if __name__ == '__main__':
    main(*sys.argv[1:])
