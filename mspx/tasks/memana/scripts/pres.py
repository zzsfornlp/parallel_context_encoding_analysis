#

# print results

import re
import os
from functools import partial
from mspx.utils import stat_analyze, zlog
from mspx.scripts.tools.print_res2 import get_all_results_from_args

# =====
# general helper

def _aggr_results(results, cmd):
    res_avg = sum(results) / len(results)
    if "full" in cmd and "avg" in cmd:
        ret = "/".join([f"{z:.2f}" for z in results]) + f";{res_avg:.2f}"
    elif "full" in cmd:
        ret = "/".join([f"{z:.2f}" for z in results])
    else:  # otherwise simply avg
        ret = f"{res_avg:.2f}"
    return ret

def f_entry(gdata, cmd: str, key: str):
    if len(gdata) == 0:
        return "N/A"
    results = [float(z[1][key]) for z in gdata]
    ret = _aggr_results(results, cmd)
    return ret

# =====
# for LM

def f_label0(dict_key):
    _zlfilter = os.environ.get("ZLFILTER")
    if _zlfilter and _zlfilter not in dict_key["line"]:
        return None
    if dict_key['AAA'] == "":
        ppp = 0
    elif dict_key['AAA'] == "D":
        ppp = "D"
    else:
        ppp = 1
    if "sel_topk:" in dict_key["line"]:
        sel_topk = int(dict_key["line"].split("sel_topk:")[1].split()[0])
    else:
        sel_topk = 0
    # xlabel = f"D={get_source(dict_key)},Seg={int(dict_key['SS']):05d},Fix={int(dict_key['PPP']):04d},S={sel_topk},P={ppp}"
    xlabel = f"Seg={int(dict_key['SS']):05d},S={sel_topk},P={ppp}"
    if "BBB" in dict_key and dict_key["BBB"].strip():
        xlabel += f",BBB={dict_key['BBB']}"
    ylabel = f"{int(dict_key['SP']):03d}"
    if "PPP" in dict_key:
        ylabel = ylabel + f",{float(dict_key['PPP'])}"
    return xlabel, ylabel

def get_source(d):
    data = "UNK"
    for cand_data in ["pg19", "proof_pile"]:
        if cand_data in d["line"]:
            data = cand_data
            break
    return data

def get_essential_data(d):
    ret = {k: v for k, v in d.items() if k in ["SS", "PPP", "SP", "BBB"]}
    ret['data'] = get_source(d)
    return ret

def _special_aug(results, key: str):
    results0 = results.copy()
    for k, v in results0:
        if k['AAA']:  # check diff
            k2 = k.copy()
            k2['AAA'] = ""
            for k3, v3 in results0:
                if get_essential_data(k3) == get_essential_data(k2):
                    k2['AAA'] = 'D'
                    v3 = v3.copy()
                    v3[key] = float(v3[key]) - float(v[key])
                    results.append((k2, v3))
                    break
# =====

# =====
# for helmet

def f_label1(dict_key):
    task = re.search(r"RUN with ([a-zA-Z0-9_]*)", dict_key["line"]).group(1)
    if dict_key['AAA'] == "":
        ppp = 0
    elif dict_key['AAA'] == "D":
        ppp = "D"
    else:
        ppp = 1
    if "sel_topk:" in dict_key["line"]:
        sel_topk = int(dict_key["line"].split("sel_topk:")[1].split()[0])
    else:
        sel_topk = 0
    xlabel = f"T={task},S={sel_topk},P={ppp}"
    if "SA_EXTRA" in dict_key["line"]:
        sa_extra = dict_key["line"].split("SA_EXTRA", 1)[1].split("=", 1)[1].strip()
        xlabel = f"{xlabel},SA={sa_extra}"
    ylabel = f"{int(dict_key['SP']):03d}"
    if "PPP" in dict_key:
        ylabel = ylabel + f",{float(dict_key['PPP'])}"
    return xlabel, ylabel

# =====

def main(args):
    conf, all_results = get_all_results_from_args(args)
    # --
    if conf.input_type.startswith("helmet"):
        _key = ([z2 for z1, z2 in [("ent", "ENTROPY")] if z1 in conf.input_type] + ["zres"])[0]
        df = stat_analyze(all_results, f_label1, partial(f_entry, cmd=conf.input_type, key=_key))
    elif conf.input_type.startswith("lm"):  # lm
        _key = ([z2 for z1, z2 in [("ent", "ENTROPY")] if z1 in conf.input_type] + ["PPL"])[0]
        # _special_aug(all_results, _key)
        df = stat_analyze(all_results, f_label0, partial(f_entry, cmd=conf.input_type, key=_key))
    else:
        df = None
    zlog(df.to_string())
    breakpoint()

# --
# python3 -m mspx.tasks.memana.scripts.pres "pat_key:RUN with" "pat_value:Finished LM"
if __name__ == '__main__':
    import sys
    main(sys.argv[1:])

# examples
"""
grep -E "RUN|Finished LM" _log_lm241016_mistralai_Mistral-7B-v0.3 | ZLFILTER=pg19 python3 -m mspx.tasks.memana.scripts.pres "pat_key:RUN with" "pat_value:Finished LM"
grep -E "RUN|Final results are" _log_lm241110_meta-llama_Meta-Llama-3.1-8B-Instruct | python3 -m mspx.tasks.memana.scripts.pres "pat_key:RUN with" "pat_value:Final results are" input_type:helmet
grep -E "RUN|Final results are" _log_lm241112_meta-llama_Meta-Llama-3.1-8B_icl_8192 | python3 -m mspx.tasks.memana.scripts.pres "pat_key:ZHEADER" "pat_value:Final results are" input_type:helmet
"""
