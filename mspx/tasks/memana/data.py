#

# prepare helmet data

import json
import math
import re
import string
import unicodedata
from collections import Counter
import numpy as np
from datasets import load_dataset

from mspx.utils import zwarn

# --
# helpers
def normalize_answer(s):

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def remove_citations(sent):
    return re.sub(r"\[\d+", "", re.sub(r" \[\d+", "", sent)).replace(" |", "").replace("]", "")

def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall

def drqa_normalize(text):
    """Resolve different type of unicode encodings."""
    return unicodedata.normalize('NFD', text)

def drqa_exact_match_score(prediction, ground_truth):
    """Check if the prediction is a (soft) exact match with the ground truth."""
    return normalize_answer(prediction) == normalize_answer(ground_truth)

def substring_exact_match_score(prediciton, ground_truth):
    """Check if the ground truth is a (soft) exact match substring of the prediction."""
    return normalize_answer(ground_truth) in normalize_answer(prediciton)

def drqa_metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    """Given a prediction and multiple valid answers, return the score of
    the best prediction-answer_n pair given a metric function.
    """
    # ground truth could be a string or a list of strings or a list of list of strings
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]
    elif isinstance(ground_truths[0], list):
        ground_truths = [ground_truth for ground_truths_list in ground_truths for ground_truth in ground_truths_list]

    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)

# note: simply no rouge!
# r_scorer = rouge_scorer.RougeScorer(['rougeL', 'rougeLsum'], use_stemmer=True)
def calculate_metrics(prediction, answers):
    em = drqa_metric_max_over_ground_truths(drqa_exact_match_score, prediction, answers)
    f1 = drqa_metric_max_over_ground_truths(lambda x, y: f1_score(x, y)[0], prediction, answers)
    sub_em = drqa_metric_max_over_ground_truths(substring_exact_match_score, prediction, answers)

    # if isinstance(answers, str):
    #     answers = [answers]
    # elif isinstance(answers[0], list):
    #     answers = [ground_truth for ground_truths_list in answers for ground_truth in ground_truths_list]
    # rouges = [r_scorer.score(target=a, prediction=prediction) for a in answers]
    # rouge = {}
    # for k in r_scorer.rouge_types:
    #     rouge[k + "_f1"] = max([r[k].fmeasure for r in rouges])
    #     rouge[k + "_recall"] = max([r[k].recall for r in rouges])

    return {
        "exact_match": em,
        "f1": f1,
        "substring_exact_match": sub_em,
        # **rouge,
    }

def parse_output(output, prefix="Answer:"):
    def lstrip_string(s, sub):
        return re.sub(f'^{re.escape(sub)}', '', s, flags=re.IGNORECASE)
    patterns = [re.compile(f"(?:{prefix})(.*)(?:\n|$)", flags=re.IGNORECASE), re.compile(r"(?:^)(.*)(?:\n|$)")]
    for pat in patterns:
        matches = pat.search(output)
        if matches is not None:
            return lstrip_string(matches[1].strip(), prefix).strip() # 0 index includes the non-capturing group # lstrip again because for chat models sometimes it will repeat the prefix
    # if still not found, return None, but should actually never get this case...
    return None

def default_post_process(output, example):
    """
    Returns: metrics (dict) and additional info to update the original sample with (dict)
    """
    prediction = output["output"]
    answer = example["answer"]
    mets = calculate_metrics(prediction, answer)
    # we check the metrics after parsing and take the max
    parsed_pred = parse_output(prediction)
    if parsed_pred is not None:
        new_mets = calculate_metrics(parsed_pred, answer)
        mets = {k: max(v, new_mets[k]) for k, v in mets.items()}
    mets["zres"] = mets["substring_exact_match"]  # a default one!
    return mets, {"parsed_output": parsed_pred}
# --

def load_qa(dataset, path):
    if path.endswith(".json"):
        data = load_dataset("json", data_files=path, field="data")["train"]
    elif path.endswith(".jsonl"):
        data = load_dataset("json", data_files=path)["train"]
    else:
        data = None

    # popularity filtering for popqa
    if "popqa" in dataset:
        popularity_threshold = float(dataset.split("_")[-1])
        data = data.filter(lambda x: math.log10(x['s_pop']) < popularity_threshold)

    passage_template = "Document (Title: {title}): {text}\n\n"
    final_data = []
    for sample in data:
        passages = [passage_template.format(**c) for c in sample['ctxs']]
        _query = f"Use the given documents to write a concise and short answer to the question. Write your answer in the following format:\nAnswer: [answer]\n\nQuestion: {sample['question']}\nAnswer:"
        final_data.append({"context": passages, "query": _query, "answer": sample["answers"]})
    return {"data": final_data, "post_process": default_post_process}

def load_json_kv(dataset, path):
    if path.endswith(".json"):
        data = load_dataset("json", data_files=path, field="data")["train"]
    elif path.endswith(".jsonl"):
        data = load_dataset("json", data_files=path)["train"]
    else:
        data = None

    final_data = []
    for sample in data:
        # parse context
        json_data = json.loads(sample["context"].split("\n", 1)[1])
        assert len(json_data) == sample["num_kvs"]
        contexts = [f'    "{a}": "{b}",\n' for a, b in json_data.items()]  # ignore the brackets '{}' for simplicity
        _oracle_idxes = [_idx for _idx, _ss in enumerate(contexts) if sample['question'] in _ss]
        assert len(_oracle_idxes) == 1, f"Get multiple keys {_oracle_idxes}"
        _query = f"}}\n\nExtract the value corresponding to the specified key in the JSON object above.\n\nKey: {sample['question']}\nCorresponding value:"
        final_data.append({"context": contexts, "oracle_idxes": _oracle_idxes, "query": _query, "answer": sample["answer"]})

    def post_process(output, example):
        prediction = output["output"]
        answer = example["answer"]
        mets = calculate_metrics(prediction, answer)
        # we don't really need to parse because we ues substring em, but could be nice to see how precise the model is
        parsed_pred = parse_output(prediction, "Corresponding value:")
        new_mets = calculate_metrics(parsed_pred, answer)
        mets = {k: max(v, new_mets[k]) for k, v in mets.items()}
        mets["zres"] = mets["substring_exact_match"]
        return mets, {"parsed_output": parsed_pred}

    return {"data": final_data, "post_process": post_process}


def load_icl(dataset, path):
    shot = int(dataset.split("shot")[0].split("_")[-1])
    _rand = np.random.RandomState(42)

    if "trec_fine" in dataset.lower():
        train_data = load_dataset("CogComp/trec", trust_remote_code=True)["train"]
        test_data = load_dataset("CogComp/trec", trust_remote_code=True)["test"]
        id2label = train_data.features['fine_label'].names
        text_field = "text"
        label_field = "fine_label"
        num_labels = 50
    elif "trec_coarse" in dataset.lower():
        train_data = load_dataset("CogComp/trec", trust_remote_code=True)["train"]
        test_data = load_dataset("CogComp/trec", trust_remote_code=True)["test"]
        id2label = train_data.features['coarse_label'].names
        text_field = "text"
        label_field = "coarse_label"
        num_labels = 6
    elif "banking77" in dataset.lower():
        train_data = load_dataset("PolyAI/banking77", trust_remote_code=True)["train"]
        test_data = load_dataset("PolyAI/banking77", trust_remote_code=True)["test"]
        id2label = train_data.features["label"].names
        id2label = {i: id2label[i] for i in range(len(id2label))}
        text_field = "text"
        label_field = "label"
        num_labels = 77
    elif "clinic150" in dataset.lower():
        train_data = load_dataset("clinc_oos", "plus")["train"]
        test_data = load_dataset("clinc_oos", "plus")["validation"]
        id2label = train_data.features["intent"].names
        text_field = "text"
        label_field = "intent"
        num_labels = 151
    elif "nlu" in dataset.lower():
        data = load_dataset("xingkunliuxtracta/nlu_evaluation_data", trust_remote_code=True)["train"]
        id2label = data.features["label"].names
        data = data.train_test_split(test_size=0.1, seed=42)
        train_data = data["train"]
        test_data = data["test"]
        text_field = "text"
        label_field = "label"
        num_labels = 68
    else:
        raise NotImplementedError(f"Unknown ICL dataset")

    def balance_labels(data, shots):
        # for each data point, we are going to sample a random set of demos with balanced labels
        # there are two places where randomness is involved: the selection of the demos and the final shuffle
        label_mapping = {x[label_field]: [] for x in data}
        for x in data:
            label_mapping[x[label_field]].append(x)
        # rearrange the data such that every label has the same number of samples
        # they are also in consecutive sets with random order in each set
        num_rounds = math.ceil(shots / len(label_mapping))  # how many rounds of balanced data?
        new_data = [[] for _ in range(num_rounds)]
        for _, samples in label_mapping.items():
            indices = list(range(len(samples)))
            _rand.shuffle(indices)
            while len(indices) < num_rounds:
                tmp_indices = list(range(len(samples)))
                _rand.shuffle(tmp_indices)
                indices += tmp_indices
            for i, idx in enumerate(indices[:num_rounds]):
                new_data[i].append(samples[idx])
        # for i in range(len(new_data)):
        #     _rand.shuffle(new_data[i])
        # new_data = [item for sublist in new_data for item in sublist][:shots]
        return new_data

    item_template = "{text}\nlabel: {label}\n\n"

    # note: simply keep the same data groups and do group shuffling
    if "balance" in dataset:
        demos0 = balance_labels(train_data, shot)
    else:
        demos0 = []
        while len(demos0) < shot:
            demos0 += list(_rand.choice(train_data, min(len(train_data), shot - len(demos0)), replace=False))

    def preprocess(sample):
        if "balance" in dataset:
            for i in range(len(demos0)):
                _rand.shuffle(demos0[i])
            demos = [item for sublist in demos0 for item in sublist][:shot]
        else:
            demos = demos0.copy()
            _rand.shuffle(demos)
        if "natural_label" in dataset:
            label_mapping = [id2label[i] for i in range(num_labels)]
        else:  # we map the labels to a random integer
            label_mapping = list(range(num_labels))
            _rand.shuffle(label_mapping)
        # --
        context = [item_template.format(text=selected_item[text_field], label=str(label_mapping[int(selected_item[label_field])])) for selected_item in demos]
        _query = f"Use the provided mapping from the text to label to assign a label to the text. Only output \"label: {{label}}\" and nothing else.\n\n{sample[text_field]}\nlabel:"
        return {"context": context, "query": _query, "answer": str(label_mapping[int(sample[label_field])])}

    final_data = [preprocess(z) for z in test_data]

    def post_process(output, example):
        prediction = output["output"]
        answer = example["answer"]
        prediction = parse_output(prediction, "label:")
        mets = calculate_metrics(prediction, answer)
        mets["zres"] = mets["exact_match"]
        return mets, {"parsed_output": prediction}

    return {"data": final_data, "post_process": post_process}


def load_ruler(dataset, path):
    data = load_dataset("json", data_files=path)["train"]

    if "niah_mk_2" in dataset:
        _query0 = "A special magic number is hidden within the above text. The special magic number for {query} mentioned in the provided text is"
    elif "niah_mk_3" in dataset:
        _query0 = "A special magic uuid is hidden within the above text. The special magic uuid for {query} mentioned in the provided text is"
    elif "niah_mv" in dataset:
        _query0 = "Some special magic numbers are hidden within the above text. The special magic numbers for {query} mentioned in the provided text are"
    else:
        raise NotImplementedError(f"Unknown ruler dataset {dataset}")

    def process_example(example):
        context = [z+"." for z in example["context"].split(".") if z.strip()]  # simply split by "."
        _query = "\n\n" + _query0.format(query=example["query"])
        _oracle_idxes = [_idx for _idx, _ss in enumerate(context) if any(zz in _ss for zz in example["answer"])]
        return {"context": context, "oracle_idxes": _oracle_idxes, "query": _query, "answer": example["answer"]}

    final_data = [process_example(z) for z in data]

    def post_process(output, example):
        # we don't do any parsing since we are only checking for substring exact match
        prediction = output["output"]
        answer = example["answer"]
        recall = sum([a.lower() in prediction.lower() for a in answer]) / len(answer)
        mets = {"ruler_recall": recall, "zres": recall}
        return mets, {"parsed_output": prediction}

    return {"data": final_data, "post_process": post_process if "qa" not in dataset else default_post_process}

# --
def load_data(dataset, path):
    if "popqa" in dataset:
        data = load_qa(dataset, path)
    elif any([x in dataset for x in ["nq", "hotpotqa", "triviaqa"]]):
        data = load_qa(dataset, path)
    elif dataset == "json_kv":
        data = load_json_kv(None, path)
    elif "icl" in dataset:
        data = load_icl(dataset, None)
    elif "ruler" in dataset:
        data = load_ruler(dataset, path)
    else:
        raise ValueError(f"Unknown dataset {dataset}")
    return data

# --
DATA_ITEMS = {
    # 4K
    "icl_4096": [{"f": load_icl, "dataset": z, "path": None, "max_gen_length": 20} for z in "icl_trec_coarse_200shot_balance,icl_trec_fine_200shot_balance,icl_banking77_180shot_balance,icl_clinic150_220shot_balance,icl_nlu_255shot_balance".split(",")],
    "rag_4096": [{"f": load_qa, "dataset": "kilt_nq", "path": "data/kilt/nq-dev-multikilt_1000_k20_dep6.jsonl", "max_gen_length": 20}, {"f": load_qa, "dataset": "kilt_triviaqa", "path": "data/kilt/triviaqa-dev-multikilt_1000_k20_dep6.jsonl", "max_gen_length": 20}, {"f": load_qa, "dataset": "kilt_hotpotqa", "path": "data/kilt/hotpotqa-dev-multikilt_1000_k20_dep3.jsonl", "max_gen_length": 20}, {"f": load_qa, "dataset": "kilt_popqa_3", "path": "data/kilt/popqa_test_1000_k20_dep6.jsonl", "max_gen_length": 20}],
    "recall_4096": [{"f": load_ruler, "dataset": a, "path": f"data/ruler/{b}/validation_4096.jsonl", "max_gen_length": (100 if a=="ruler_niah_mk_3" else 50)} for a, b in zip(["ruler_niah_mk_2", "ruler_niah_mk_3", "ruler_niah_mv"], ["niah_multikey_2", "niah_multikey_3", "niah_multivalue"])] + [{"f": load_json_kv, "dataset": "json_kv", "path": "data/json_kv/test_k50_dep6.jsonl", "max_gen_length": 100}],
    # 8K
    "icl_8192": [{"f": load_icl, "dataset": z, "path": None, "max_gen_length": 20} for z in "icl_trec_coarse_400shot_balance,icl_trec_fine_400shot_balance,icl_banking77_360shot_balance,icl_clinic150_440shot_balance,icl_nlu_510shot_balance".split(",")],
    "rag_8192": [{"f": load_qa, "dataset": "kilt_nq", "path": "data/kilt/nq-dev-multikilt_1000_k50_dep6.jsonl", "max_gen_length": 20}, {"f": load_qa, "dataset": "kilt_triviaqa", "path": "data/kilt/triviaqa-dev-multikilt_1000_k50_dep6.jsonl", "max_gen_length": 20}, {"f": load_qa, "dataset": "kilt_hotpotqa", "path": "data/kilt/hotpotqa-dev-multikilt_1000_k50_dep3.jsonl", "max_gen_length": 20}, {"f": load_qa, "dataset": "kilt_popqa_3", "path": "data/kilt/popqa_test_1000_k50_dep6.jsonl", "max_gen_length": 20}],
    "recall_8192": [{"f": load_ruler, "dataset": a, "path": f"data/ruler/{b}/validation_8192.jsonl", "max_gen_length": (100 if a=="ruler_niah_mk_3" else 50)} for a, b in zip(["ruler_niah_mk_2", "ruler_niah_mk_3", "ruler_niah_mv"], ["niah_multikey_2", "niah_multikey_3", "niah_multivalue"])] + [{"f": load_json_kv, "dataset": "json_kv", "path": "data/json_kv/test_k105_dep6.jsonl", "max_gen_length": 100}],
    # 16K
    "icl_16K": [{"f": load_icl, "dataset": z, "path": None, "max_gen_length": 20} for z in "icl_trec_coarse_800shot_balance,icl_trec_fine_800shot_balance,icl_banking77_720shot_balance,icl_clinic150_880shot_balance,icl_nlu_1020shot_balance".split(",")],
    "rag_16K": [{"f": load_qa, "dataset": "kilt_nq", "path": "data/kilt/nq-dev-multikilt_1000_k105_dep6.jsonl", "max_gen_length": 20}, {"f": load_qa, "dataset": "kilt_triviaqa", "path": "data/kilt/triviaqa-dev-multikilt_1000_k105_dep6.jsonl", "max_gen_length": 20}, {"f": load_qa, "dataset": "kilt_hotpotqa", "path": "data/kilt/hotpotqa-dev-multikilt_1000_k105_dep3.jsonl", "max_gen_length": 20}, {"f": load_qa, "dataset": "kilt_popqa_3", "path": "data/kilt/popqa_test_1000_k105_dep6.jsonl", "max_gen_length": 20}],
    "recall_16K": [{"f": load_ruler, "dataset": a, "path": f"data/ruler/{b}/validation_16384.jsonl", "max_gen_length": (100 if a=="ruler_niah_mk_3" else 50)} for a, b in zip(["ruler_niah_mk_2", "ruler_niah_mk_3", "ruler_niah_mv"], ["niah_multikey_2", "niah_multikey_3", "niah_multivalue"])] + [{"f": load_json_kv, "dataset": "json_kv", "path": "data/json_kv/test_k220_dep6.jsonl", "max_gen_length": 100}],
    # --
    # testing
    "debug_rag": [{"f": load_qa, "dataset": "kilt_hotpotqa", "path": "data/kilt/hotpotqa-dev-multikilt_1000_k50_dep3.jsonl", "max_gen_length": 20}],
    "debug_icl": [{"f": load_icl, "dataset": "icl_trec_fine_400shot_balance", "path": None, "max_gen_length": 20}],
    "debug_json": [{"f": load_json_kv, "dataset": "json_kv", "path": "data/json_kv/test_k50_dep6.jsonl", "max_gen_length": 100}],
}

# Counter({'context': 200000, 'inst': 500, 'context_N=400': 500})
# Counter({'context': 200000, 'inst': 500, 'context_N=400': 500})
# Counter({'context': 1108800, 'inst': 3080, 'context_N=360': 3080})
# Counter({'context': 1364000, 'inst': 3100, 'context_N=440': 3100})
# Counter({'context': 1311720, 'inst': 2572, 'context_N=510': 2572})
# Counter({'context': 297900, 'inst': 5958, 'context_N=50': 5958})
# Counter({'context': 292800, 'inst': 5856, 'context_N=50': 5856})
# Counter({'context': 148050, 'inst': 2961, 'context_N=50': 2961})
# Counter({'context': 44100, 'inst': 882, 'context_N=50': 882})
# Counter({'context': 32600, 'inst': 100, 'context_N=326': 100})
# Counter({'context': 7600, 'inst': 100, 'context_N=76': 100})
# Counter({'context': 35392, 'inst': 100, 'context_N=354': 92, 'context_N=353': 8})
# Counter({'context': 63000, 'inst': 600, 'context_N=105': 600})

# --
# for debug
def do_debug_test():
    for k in ["icl_8192", "qa_8192", "recall_8192"]:
        vs = DATA_ITEMS[k]
        for vv in vs:
            print(f"Run with {vv}")
            kwargs = vv.copy()
            del kwargs["f"]
            data = vv["f"](**kwargs)["data"]
            cc = Counter()
            for d in data:
                _num_context = len(d["context"])
                cc['inst'] += 1
                cc['context'] += _num_context
                cc[f'context_N={_num_context}'] += 1
            print(cc)
            # breakpoint()

# python -mpdb -m mspx.tasks.memana.data
if __name__ == '__main__':
    do_debug_test()
