#

# about files

__all__ = [
    "zopen", "WithWrapper", "zopen_withwrapper", "dir_msp",
]

from typing import IO, Union, Callable, Iterable
import os
import sys
import io
import glob

# --
# helper for zstd

class ZstdReader:
    def __init__(self, filename, encoding="utf-8"):
        import zstandard as zstd
        self.fd = open(filename, 'rb')
        dctx = zstd.ZstdDecompressor()
        self.reader = dctx.stream_reader(self.fd)
        self.io_wrapper = io.TextIOWrapper(self.reader, encoding=encoding)

    def __enter__(self):
        return self.io_wrapper

    def __exit__(self, *a):
        self.fd.close()
        return False

class ZstdWriter:
    def __init__(self, filename, encoding="utf-8"):
        import zstandard as zstd
        self.fd = open(filename, 'wb')
        ctx = zstd.ZstdCompressor()
        self.writer = ctx.stream_writer(self.fd)
        self.io_wrapper = io.TextIOWrapper(self.writer, encoding=encoding)

    def __enter__(self):
        return self.io_wrapper

    def __exit__(self, *a):
        import zstandard as zstd
        self.io_wrapper.flush()
        self.writer.flush(zstd.FLUSH_FRAME)
        self.fd.close()
        return False

def open_zstd(filename, mode='rb', **kwargs):
    if 'w' in mode:
        return ZstdWriter(filename, **kwargs)
    else:
        return ZstdReader(filename, **kwargs)
# --

# open various kinds of files
def zopen(filename: str, mode='r', encoding="utf-8", check_zip=True):
    suffix = "" if ('b' in mode) else "t"
    if 'b' in mode:
        encoding = None
    fd = None
    if check_zip:
        if filename.endswith('.gz'):
            import gzip
            fd = gzip.open(filename, mode+suffix, encoding=encoding)
        elif filename.endswith('.bz2'):
            import bz2
            fd = bz2.open(filename, mode+suffix, encoding=encoding)
        elif filename.endswith(".zst"):
            fd = open_zstd(filename, mode+suffix, encoding=encoding)
    if fd is None:
        return open(filename, mode, encoding=encoding)
    else:
        return fd

# a simple wrapper class for with expression
class WithWrapper:
    def __init__(self, f_start: Callable = None, f_end: Callable = None, item=None):
        self.f_start = f_start
        self.f_end = f_end
        self.item: object = item

    def __enter__(self):
        if self.f_start is not None:
            self.f_start()
        if self.item is not None and hasattr(self.item, "__enter__"):
            self.item.__enter__()
        # return self if self.item is None else self.item
        return self.item

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.item is not None and hasattr(self.item, "__exit__"):
            self.item.__exit__()
        if self.f_end is not None:
            self.f_end()

# open (possibly with fd)
def zopen_withwrapper(fd_or_path: Union[str, IO], mode='r', empty_std=False, **kwargs):
    if empty_std and fd_or_path == '':
        fd_or_path = sys.stdout if ('w' in mode) else sys.stdin
    if isinstance(fd_or_path, str) and fd_or_path:
        return zopen(fd_or_path, mode=mode, **kwargs)
    else:
        # assert isinstance(fd_or_path, IO)
        return WithWrapper(None, None, fd_or_path)

# get msp's directory
def dir_msp(absolute=False):
    dir_name = os.path.dirname(os.path.abspath(__file__))  # msp?/utils
    dir_name2 = os.path.join(dir_name, "..")  # msp?
    if absolute:
        dir_name2 = os.path.abspath(dir_name2)
    return dir_name2
