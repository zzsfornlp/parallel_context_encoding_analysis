#

# about system information

__all__ = [
    "system", "get_statm", "get_sysinfo",
    "zglob", "zglob1", "mkdir_p", "auto_mkdir", "resymlink", "auto_prep_path",
]

import sys, os, subprocess, traceback, glob, platform
from typing import Iterable, Union, List
from .log import zopen, zlog, zcheck, zwarn

# performing system CMD
def system(cmd: str, pp=False, ass=False, popen=False, return_code=False):
    if pp:
        zlog(f"Executing cmd: {cmd}")
    if popen:
        try:
            tmp_out = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, executable='/bin/bash')
            n = 0
            output = str(tmp_out.decode())  # byte->str
        except subprocess.CalledProcessError as grepexc:
            n = grepexc.returncode
            output = grepexc.output
        # p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
        # n = p.wait()
        # output = str(p.stdout.read().decode())      # byte -> str
    else:
        n = os.system(cmd)
        output = None
    if pp:
        zlog(f"Output is: {output}")
    if ass:
        assert n == 0, f"Executing previous cmd returns error {n}"
    if return_code:
        return output, n
    else:
        return output
    # --

# get mem info
def get_statm():
    with zopen("/proc/self/statm") as f:
        rss = (f.read().split())        # strange!! readline-nope, read-ok
        mem0 = str(int(rss[1])*4//1024) + "MiB"
    try:
        line = system("nvidia-smi | grep -E '%s.*MiB'" % os.getpid())
        mem1 = line[-1].split()[-2]
    except:
        mem1 = "0MiB"
    return mem0, mem1

# get sys info
def get_sysinfo(ret_str=True, get_gpu_info=False):
    ret = {'uname': platform.uname(), 'gpu': system("nvidia-smi", popen=True) if get_gpu_info else None}
    if ret_str:
        ret = f"#== Sysinfo: {ret['uname']}\n{ret['gpu']}"
    return ret

# traceback info
# traceback.format_exc(limit=None)
# traceback.format_stack(limit=None)

def zglob(paths: Union[Iterable[str], str], check_prefix="..", check_iter=0, sorted=True, only_one=False, err_act='warn'):
    if isinstance(paths, str):
        paths = [paths]
    ret = []
    for pathname in paths:
        if pathname == "":
            files = []
        else:
            if pathname.startswith("__"):  # note: special semantics!!
                pathname = pathname[2:]
                check_iter = max(check_iter, 10)
            files = glob.glob(pathname)
            if len(check_prefix) > 0:
                while len(files) == 0 and check_iter > 0:
                    pathname = os.path.join(check_prefix, pathname)
                    files = glob.glob(pathname)
                    check_iter -= 1  # limit for checking
            if sorted:  # only sort things inside
                files.sort()
        ret.extend(files)
        if len(files) == 0:
            zcheck(False, s=f'Cannot find any files with {pathname}', err_act=err_act)
    if only_one:
        if len(ret) != 1:
            zcheck(False, s=f'Get multiple files when "only_one": {paths} -> {ret}', err_act=err_act)
        return ret[0]
    else:
        return ret

# shortcut
zglob1 = lambda *args, **kwargs: zglob(*args, only_one=True, **kwargs)

# mkdir -p path
def mkdir_p(path: str, err_act='warn', **kwargs):
    if os.path.exists(path):
        if os.path.isdir(path):
            return True
        zcheck(False, f"Failed mkdir: {path} exists and is not dir!", err_act=err_act)
        return False
    else:
        # os.mkdir(path)
        os.makedirs(path, **kwargs)
        return True

# auto mkdir
def auto_mkdir(path: str, **kwargs):
    dir_name = os.path.dirname(path)
    if dir_name and not os.path.exists(dir_name):
        mkdir_p(dir_name, **kwargs)

# relink file: delete if existing
def resymlink(src, dst):
    if os.path.islink(dst):
        os.unlink(dst)
    os.symlink(src, dst)

# auto prepare paths
def auto_prep_path(path_dir: Union[str, List[str]] = None, path_file: Union[str, List[str]] = None, **kwargs):
    assert (path_dir is None) != (path_file is None)
    _trg_dir = path_dir  # target dir to make
    if _trg_dir is None:
        if isinstance(path_file, list):
            path_file = os.path.join(*path_file) if path_file else ""
        _trg_dir = os.path.dirname(path_file)  # get its dirname!
    else:
        if isinstance(path_dir, list):
            path_dir = os.path.join(*path_dir) if path_dir else ""
        _trg_dir = path_dir
    if _trg_dir:
        if os.path.exists(_trg_dir):
            assert os.path.isdir(_trg_dir), f"Failed mkdir: {_trg_dir} exists and is not dir!"
        else:
            os.makedirs(_trg_dir, **kwargs)
    return path_dir if path_file is None else path_file
