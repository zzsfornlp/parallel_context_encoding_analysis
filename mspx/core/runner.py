#

# base runner
# main running processes, such as training and testing (simplified version)

__all__ = [
    "RunnerConf", "MyRunner"
]

import os
import time
import torch
import math
from collections import OrderedDict
from functools import partial
from contextlib import nullcontext
from tqdm.auto import tqdm
from typing import Dict, Union, List
import numpy as np
from mspx.utils import zlog, zwarn, Conf, Configurable, StatRecorder, ZResult, ZHelper, Timer, Constants, Logger, mkdir_p, default_json_serializer, zglob, zglob1, Random
from mspx.nn import BK
from .helper import *

# --
class RunnerConf(Conf):
    def __init__(self):
        # optim
        self.learning_rate = 0.00025  # max-lr
        self.weight_decay = 0.1  # weight-decay for AdamW
        self.gradient_clipping = 1.0  # gradient clipping
        self.adam_betas = [0.9, 0.95]
        # lrate-scheduler
        self.lr_warmup_ratio = 0.1  # warmup steps = ratio * max_steps
        self.lr_decay_ratio = 1.0  # total decay steps = ratio * max_steps
        self.min_lr_ratio = 0.1  # min-lr == ratio * lrate
        # train
        self.accu_batch = 1  # how many fb count as one update
        self.min_uidx = 0  # min updated (before early stop)
        self.max_uidx = 1000 * 10  # max update
        self.max_eidx = Constants.INT_PRAC_MAX  # num of epoch
        self.train_ckp = ""  # loading for training
        self.resume_from_ckp = '_AUTO_'  # resume training?
        self.report_ufreq = 10  # report every ? steps
        # valid & save(new)
        self.valid_ufreq = 1000  # do valid every this udix
        self.valid_first = False  # valid once at the very start
        self.valid_start_uidx = 0  # do valid >=this
        self.valid_nbatch = -1  # number of batches to valid
        # model save
        self.model_filter_keys = [""]  # filters for saving names (also support RE)
        self.model_save_prefix = "zmodel"  # as those allowed by "parse_save_load_name"
        self.model_save_suffix_curr = ""
        self.model_save_suffix_best = ""
        self.save_ufreq = -1  # save special points
        self.save_ustart = 0  # saving starting from ??
        self.save_latest_ufreq = -1  # save latest state
        self.save_latest_name = "latest"
        # record best & save cidx
        self.record_best_start_cidx = 0  # starting from where to record best
        # misc
        self.print_progress = False  # progress bar
        self.debug_print_step = False  # printing for debugging
        # testing
        self.test_ckp = ""  # loading for testing
        self.test_with_dropout = False
        # self.test_no_grad = True
        # self.test_report_interval = 1000  # report for testing
        # --
        # deepspeed
        self.ds_conf = DsConf()

class DsConf(Conf):
    def __init__(self):
        self.deepspeed = False  # Enable DeepSpeed?
        self.deepspeed_config = None  # DeepSpeed json configuration file
        self.local_rank = 0
        # --
        self.ds_save_overwrite = True  # otherwise save each point as "global_step*"
        self.ds_filename = "_ds.json"  # file to write
        self.ds_base = ""  # base one
        self.ds_extras = {}  # extra ones: {name -> value}

    def setup_config(self, base_ds_extras=None):
        if self.deepspeed_config:
            return  # no need since already provided as an external file!
        from .ds_confs import LIB_DS
        from copy import deepcopy
        conf = deepcopy(LIB_DS[self.ds_base])
        # --
        ds_extras = base_ds_extras.copy() if base_ds_extras else {}
        ds_extras.update(self.ds_extras)
        for name, value in ds_extras.items():
            path = name.split(".")
            curr = conf
            for _piece in path[:-1]:
                if _piece not in curr:  # create one if not existing
                    curr[_piece] = {}
                curr = curr[_piece]
            _piece = path[-1]
            if _piece in curr:
                curr[_piece] = Conf.typed_convert(value, curr[_piece])
            else:
                curr[_piece] = value  # directly assign!
        # --
        zlog(f"Setup config to {self.ds_filename} with base={self.ds_base} extras={ds_extras}")
        default_json_serializer.to_file(conf, self.ds_filename)
        self.deepspeed_config = self.ds_filename
        self.local_rank = BK.get_rank_info()[0]  # reset local rank!!

class MyRunner(Configurable):
    def __init__(self, conf: RunnerConf):
        super().__init__(conf)
        conf: RunnerConf = self.conf
        # --

    def get_lr_schedular(self, optimizer):
        conf: RunnerConf = self.conf
        # --
        base_lr = 1.
        min_lr = base_lr * conf.min_lr_ratio
        warmup_steps, decay_steps = int(conf.lr_warmup_ratio * conf.max_uidx), int(conf.lr_decay_ratio * conf.max_uidx)
        # --
        def get_lr(it: int):
            # 1) linear warmup for warmup_iters steps
            if it < warmup_steps:
                return base_lr * it / warmup_steps
            # 2) if it > lr_decay_iters, return min learning rate
            if it > decay_steps:
                return min_lr
            # 3) in between, use cosine decay down to min learning rate
            decay_ratio = (it - warmup_steps) / (decay_steps - warmup_steps)
            assert 0 <= decay_ratio <= 1
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
            return min_lr + coeff * (base_lr - min_lr)
        # --
        from torch.optim.lr_scheduler import LambdaLR
        zlog(f"Obtain LambdaLR with {warmup_steps}-{decay_steps}-{conf.max_uidx}({min_lr})")
        return LambdaLR(optimizer, get_lr)

    def prepare_training(self, model):
        conf: RunnerConf = self.conf
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        # try:
        #     from apex.optimizers import FusedAdam
        #     optimizer = FusedAdam(trainable_params, lr=conf.learning_rate, betas=conf.adam_betas, weight_decay=conf.weight_decay)
        # except:
        optimizer = torch.optim.AdamW(trainable_params, lr=conf.learning_rate, betas=conf.adam_betas, weight_decay=conf.weight_decay)
        lr_scheduler = self.get_lr_schedular(optimizer)
        zlog(f"Use optimizer of {optimizer}")
        return optimizer, lr_scheduler

    # helper for iter multiple dataset
    def _iter_train_data(self, datasets, tp):
        _gen = Random.get_generator("_iter_data")  # use specific one!
        arr_w = np.asarray([d.sample_weight for d in datasets], dtype=np.float32)
        arr_w = arr_w / arr_w.sum()  # normalize
        _iters = [iter(z.get_dataloader()) for z in datasets]
        zlog(f"Start to iter data {datasets} with {arr_w}")
        for one_idx in Random.stream(_gen.choice, a=len(arr_w), p=arr_w):
            dd = datasets[one_idx]
            tp.update_fidx(1, dname=dd.name)
            cur_batch = None
            # fetch data
            for _ in range(2):
                try:
                    cur_batch = next(_iters[one_idx])
                    break
                except StopIteration:
                    _iters[one_idx] = iter(datasets[one_idx].get_dataloader())  # restart!
                    zlog(f"Start a new epoch of the data: {dd}", timed=True)
            if cur_batch is None:
                zwarn("Data Fetching Error (maybe empty data?)")
            else:
                yield cur_batch
        # --

    # main training
    def do_train(self, model, train_data: List, dev_data, extra_scheduled_values=None):
        conf: RunnerConf = self.conf
        LOCAL_RANK, RANK, WORLD_SIZE = BK.get_rank_info()
        _deepspeed = conf.ds_conf.deepspeed  # use deepspeed?
        scheduled_values = OrderedDict()  # add all scheduled values
        if extra_scheduled_values:
            ZHelper.update_dict(scheduled_values, extra_scheduled_values)
        # --
        # train
        if not isinstance(train_data, list):
            train_data = [train_data]
        tp = TrainingProgressRecord()
        train_recorder = StatRecorder()
        _batch_info = [(z.name, z.sample_weight, z.get_batch_info()) for z in train_data]
        zlog(f"=====> Start to run training PAR=[{LOCAL_RANK}/{RANK}/{WORLD_SIZE}]")
        zlog(f"  One batch size per device = {_batch_info}")
        zlog(f"  Total extra counts = {WORLD_SIZE} x {conf.accu_batch} = {WORLD_SIZE * conf.accu_batch} ")
        zlog(f"  Total epoch||steps = {conf.max_eidx}||{conf.max_uidx}")
        # --
        # prepare
        # train_dataloader, dev_dataloader = train_data.get_dataloader(), dev_data.get_dataloader()
        dev_dataloader = dev_data.get_dataloader() if dev_data is not None else None
        zlog(f"Model info before wrapping: {info_model(model)} ***:\n{info_trainable_parameters(model).to_string()}")
        if _deepspeed:
            import deepspeed
            # simply put the first one ...
            _extra_conf_dict = {"train_micro_batch_size_per_gpu": _batch_info[0][-1]["batch_size"], "gradient_accumulation_steps": conf.accu_batch, "gradient_clipping": conf.gradient_clipping, "optimizer.params.lr": conf.learning_rate, "optimizer.params.betas": conf.adam_betas, "optimizer.params.weight_decay": conf.weight_decay}
            _extra_conf_dict.update(BK.setup_deepspeed())
            conf.ds_conf.setup_config(_extra_conf_dict)
            model_engine, optimizer, _, _ = deepspeed.initialize(args=conf.ds_conf, model=model, lr_scheduler=self.get_lr_schedular)
            lr_scheduler = None  # not kept here!
            mixp_autocast, mixp_scaler = None, None
        else:
            model_engine = BK.wrap_model(model)
            optimizer, lr_scheduler = self.prepare_training(model_engine)
            mixp_autocast, mixp_scaler = BK.get_autocast_and_scaler()
        if mixp_autocast is None:
            mixp_autocast = nullcontext
        zlog(f"Model info after wrapping: {info_model(model_engine)} ***:\n{info_trainable_parameters(model_engine).to_string()}")
        # --
        # Only show the progress bar once on each machine.
        progress_bar = tqdm(total=conf.max_uidx, desc="Training", disable=not (BK.is_local_main_process() and conf.print_progress))
        # valid first?
        overall_info = {'last_dev_uidx': 0, 'last_dev_time': time.perf_counter(), 'last_report_time': time.perf_counter()}
        # running
        if conf.train_ckp:  # loading from training
            _load_dir = zglob1(conf.train_ckp)
            self.do_load(_load_dir, model_engine, None, None, None)
        _resume_dir = conf.resume_from_ckp
        if _resume_dir == "_AUTO_":
            _resume_dir = conf.save_latest_name
        load_dirs = zglob(_resume_dir)
        cur_batch, train_data_iter = None, self._iter_train_data(train_data, tp)  # start!
        if load_dirs:  # resume training
            load_dirs.sort(key=os.path.getctime)
            load_dir = load_dirs[-1]
            self.do_load(load_dir, model_engine, tp, optimizer, lr_scheduler)
            zlog(f"Load training progress from {load_dir} & fast-forward data-loader!")
            with Timer(info=f"Fast-forward data (f={tp.fidx})", print_date=True):
                for _ in range(tp.fidx):
                    # _, data_iter = self.fetch_data(train_dataloader, data_iter)
                    _ = next(train_data_iter)
            progress_bar.update(tp.uidx)
            overall_info['last_dev_uidx'] = tp.uidx
        else:
            if conf.valid_first:  # valid before training
                self.do_dev(model_engine, dev_dataloader, tp, train_recorder, overall_info)
            self.adjust_scheduled_values(tp, scheduled_values, optimizer)  # once before train
        loss_factor = 1. / conf.accu_batch
        # note: here we need to mix with backward/update, therefore handling it at the outside (different to testing)
        next_fetch_new = True  # whether fetch next data!
        while tp.eidx < conf.max_eidx and tp.uidx < conf.max_uidx:
            # get more data
            with train_recorder.go('fetch'):  # also record this!
                if next_fetch_new:  # only if fetch new data!
                    # cur_batch, data_iter = self.fetch_data(train_dataloader, data_iter, tp=tp)
                    cur_batch = next(train_data_iter)
            # going!
            model_engine.train()
            with train_recorder.go('fb'):
                with mixp_autocast():  # possible autocast env
                    one_outputs, next_fetch_new = self.run_train(model_engine, cur_batch)
                loss = one_outputs.loss
                if loss is not None:  # if we have loss!
                    if _deepspeed:
                        model_engine.backward(loss)  # let ds handle accu_batch!
                    else:
                        if loss_factor != 1.:
                            loss = loss * loss_factor
                        if mixp_scaler is not None:
                            mixp_scaler.scale(loss).backward()
                        else:
                            loss.backward()
                fb_res = one_outputs.res
                fb_res['fb'] = 1
                train_recorder.record(fb_res)
                self.run_post_fb(model_engine)
            # tp.update_iidx(len(cur_batch))  # nope: not always list[insts]!
            # tp.update_fidx(1)
            # --
            if _deepspeed:
                _is_boundary = model_engine.is_gradient_accumulation_boundary()
                with train_recorder.go('update'):  # also record this!
                    model_engine.step()
            else:
                _is_boundary = (tp.fidx % conf.accu_batch == 0)
            if _is_boundary:  # update
                if not _deepspeed:
                    with train_recorder.go('update'):  # also record this!
                        if mixp_scaler is not None:
                            mixp_scaler.unscale_(optimizer)  # unscale before clipping
                        BK.clip_gradient_(model_engine, conf.gradient_clipping)  # gradient clipping
                        if mixp_scaler is not None:
                            mixp_scaler.step(optimizer)
                            mixp_scaler.update()
                        else:
                            optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad()
                progress_bar.update(1)
                tp.update_uidx(1)
                cur_uidx = tp.uidx
                # valid
                if conf.valid_ufreq > 0 and cur_uidx % conf.valid_ufreq == 0:
                    self.do_dev(model_engine, dev_dataloader, tp, train_recorder, overall_info)
                    self.adjust_scheduled_values(tp, scheduled_values, optimizer)  # after dev
                # save
                if conf.save_ufreq > 0 and cur_uidx >= conf.save_ustart and cur_uidx % conf.save_ufreq == 0:
                    ss2 = tp.current_suffix(brief=True)  # brief for saving
                    self.do_save(conf.model_save_prefix+ss2, model_engine, tp, optimizer, lr_scheduler)
                # save latest
                if conf.save_latest_ufreq > 0 and cur_uidx % conf.save_latest_ufreq == 0:
                    self.do_save(conf.save_latest_name, model_engine, tp, optimizer, lr_scheduler)
                # log
                if conf.report_ufreq > 0 and cur_uidx % conf.report_ufreq == 0:
                    _stamp = time.perf_counter()
                    _gap = _stamp - overall_info['last_report_time']
                    _print_res = {}  # print all results in one
                    _print_res.update(fb_res)
                    _print_res.update(lr=self.get_curr_lrates(optimizer), extra=one_outputs.extra_info, step=tp.current_suffix(), time=_gap)
                    zlog(f"Step {tp.current_suffix()} [time={_gap:.2f}s]: {_print_res}", timed=True)
                    overall_info['last_report_time'] = _stamp
            # --
        # --
        if conf.max_eidx <= 0:
            zlog("Skip training!")
        else:
            info_best = tp.info_best()
            self.do_save(conf.model_save_prefix+".final", model_engine, tp, None, None)  # final saving
            # zlog(f"zzzzzfinal: After training, the best point is: {info_best[-1].to_dict()}.", func="report")
            zlog(f"zzzzzdevfinal: {info_best[-1].to_dict(store_all_fields=True)}.")
            zlog(f"zzzzz-----: After training, the best point is: {info_best}.", func="report")
        # --
        return model_engine

    def do_dev(self, model_engine, dev_dataloader, tp, train_recorder, overall_info):
        conf: RunnerConf = self.conf
        # --
        # report & reset training stat
        if tp.uidx > 0:
            train_result = self.run_train_report(train_recorder, tp)  # first report training stat
            train_recorder.reset()  # reset training stat
        else:  # for validate_first
            train_result = ZResult()
        # dev
        ss, cur_cidx = tp.current_suffix(), tp.cidx
        # --
        zlog("")  # empty line
        with Timer(info=f"Valid {ss}", print_date=True):
            # no validation if specified
            if tp.uidx < conf.valid_start_uidx:
                zlog("No validation since not the time yet!")
                return
            # validate
            if dev_dataloader is None:  # simply use train if there are no dev
                zlog("Use training results for dev since there are no dev set provided!", func="warn")
                dev_result = train_result
            else:
                dev_result = self.do_test(model_engine, dev_dataloader, test_num_batch=conf.valid_nbatch)
            # breakpoint()
            _print_res = dict(dev_result) if dev_result else {}
            _print_res.update(step=ss)
            zlog(f"Step-DEV {ss}: {_print_res}", timed=True)  # print dev result again
            # record
            cur_record_best = (tp.cidx >= conf.record_best_start_cidx)
            if_overall_best, if_best = tp.update_checkpoint(train_result, dev_result, record_best=cur_record_best)
            # save curr & best: no save optimizer here!
            if conf.model_save_suffix_curr:
                self.do_save(conf.model_save_prefix+conf.model_save_suffix_curr, model_engine, tp, None, None)
            if if_overall_best:
                zlog("Curr is overall best " + str(tp.info_overall_best()), func="result")
            else:
                zlog("Curr not overall best, the overall best is " + str(tp.info_overall_best()), func="result")
            if if_best:
                if conf.model_save_suffix_best:
                    self.do_save(conf.model_save_prefix+conf.model_save_suffix_best, model_engine, tp, None, None)
                zlog("Curr is best: " + str(tp.info_best()), func="result")
            else:
                zlog("Curr not best, the best is " + str(tp.info_best()), func="result")
            # --
        # --
        _stamp = time.perf_counter()
        zlog(f"END dev at {time.ctime()} ||| {_stamp-overall_info['last_dev_time']:.2f} secs from last_dev.")
        overall_info.update({'last_dev_uidx': tp.uidx, 'last_dev_time': _stamp})
        zlog("")  # empty line
        Logger.get_singleton_logger().flush_cached_logs()

    def do_test(self, model_engine, dataloader, do_init=False, **kwargs):
        conf: RunnerConf = self.conf
        # --
        if conf.test_with_dropout:
            model_engine.train()  # special mode!
        else:
            model_engine.eval()  # note: remember to make it eval!
        # --
        if do_init:
            _deepspeed = conf.ds_conf.deepspeed  # use deepspeed?
            if _deepspeed:
                import deepspeed
                zwarn("Simply load deepspeed model by hand!!")  # todo(+N): enhance deepspeed!
                # model_engine = deepspeed.init_inference(model, checkpoint=(conf.test_ckp if conf.test_ckp else None))
                if conf.test_ckp:
                    _ckp = conf.test_ckp
                    if os.path.isdir(_ckp):
                        _ckp = zglob1(os.path.join(_ckp, "*.pt"))
                    state_dict = torch.load(_ckp)
                    BK.load_my_checkpoint(state_dict["module"], model_engine, quiet=False)
            else:
                if conf.test_ckp:
                    self.do_load(conf.test_ckp, model_engine, None, None, None)
        # --
        if dataloader is not None:
            test_recorder = StatRecorder(report_key='inst', report_interval=1000)
            with Timer(info=f"Test", print_date=True):
                with BK.no_grad():
                    res = self.run_test(model_engine, dataloader, test_recorder, **kwargs)
            zlog(f"Test-Info: {ZHelper.printd_str(res, sep=' ')}")
        else:
            res = {}
            zlog("Mainly for loading model!")
        # --
        return ZResult(res)

    def adjust_scheduled_values(self, tp, scheduled_values, optimizer=None):
        # adjust schedule values
        ss = tp.current_suffix()
        for one_name, one_sv in scheduled_values.items():
            if one_sv.changeable:
                one_sv.adjust_at_ckp(ss, tp, extra_info=one_name)
        # also check current lrate
        if optimizer is None:
            zwarn("Cannot check lrate: optimizer is None")
        else:
            zlog(f"Current lrates: {self.get_curr_lrates(optimizer)}")
        # --

    def get_curr_lrates(self, optimizer):
        lrates = [pg['lr'] for pg in optimizer.param_groups]
        return lrates

    # print and return train summary
    def run_train_report(self, train_recorder, tp):
        x = train_recorder.summary()
        # also report uidx_counter/iidx_counter
        zlog(f"Train-Info: {ZHelper.printd_str(x, sep=' ')} || UidxCounter={tp.uidx_counter} FidxCounter={tp.fidx_counter}")
        x['zres'] = - x.get('loss', 0.) / x.get('fb', 1)
        return ZResult(x)

    # # fetch data
    # def fetch_data(self, dataloader, data_iter, tp=None):
    #     for _ in range(2):
    #         try:
    #             cur_batch = next(data_iter)
    #             return cur_batch, data_iter
    #         except StopIteration:
    #             data_iter = iter(dataloader)  # restart!
    #             if tp is not None:
    #                 tp.update_eidx(1)
    #                 zlog(f"Start a new epoch of the data: {tp.current_suffix()}", timed=True)
    #     raise RuntimeError("Data Fetching Error (maybe empty data?)")

    # save & load
    def do_save(self, save_dir: str, model_engine, tp, optimizer, lr_scheduler):
        _ds_conf = self.conf.ds_conf
        _deepspeed = _ds_conf.deepspeed  # use deepspeed?
        _filter_keys = self.conf.model_filter_keys
        # --
        mkdir_p(save_dir, exist_ok=True)
        with Timer("Model-save"):
            if _deepspeed:
                assert _filter_keys == [] or _filter_keys == [""], f"Filter_keys not supported for this mode: {_filter_keys}"
                model_engine.save_checkpoint(save_dir, tag=("model" if _ds_conf.ds_save_overwrite else None), client_state={"tp": tp.to_dict()})
            else:
                BK.save_my_checkpoint(save_dir, model_engine, optimizer=optimizer, filter_keys=_filter_keys)
        if BK.is_main_process():
            default_json_serializer.to_file(tp, os.path.join(save_dir, "tp.json"), indent=2)
            if lr_scheduler is not None:
                default_json_serializer.to_file(lr_scheduler.state_dict(), os.path.join(save_dir, "lrs.json"), indent=2)
        zlog(f"Save model [ds={_deepspeed}] to {save_dir}")

    def do_load(self, load_dir: str, model_engine, tp, optimizer, lr_scheduler):
        _deepspeed = self.conf.ds_conf.deepspeed  # use deepspeed?
        # --
        with Timer("Model-load"):
            if _deepspeed:
                _, client_sd = model_engine.load_checkpoint(load_dir)
            else:
                BK.load_my_checkpoint(load_dir, model_engine, optimizer=optimizer)
        if tp is not None:
            tp2 = TrainingProgressRecord.create_from_file(os.path.join(load_dir, "tp.json"))
            tp.update(tp2)
        if lr_scheduler is not None:
            sd_lrs = default_json_serializer.from_file(os.path.join(load_dir, "lrs.json"))
            lr_scheduler.load_state_dict(sd_lrs)
        zlog(f"Load model [ds={_deepspeed}] from {load_dir}")

    # model & task specific
    def run_train(self, model, data):
        raise NotImplementedError()

    def run_post_fb(self, model):
        raise NotImplementedError()

    # testing
    def run_test(self, model, data, test_recorder, **kwargs):
        raise NotImplementedError()
