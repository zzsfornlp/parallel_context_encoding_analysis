#

# multiprocessing task runner

__all__ = [
    "MyTaskRunnerConf", "MyTaskRunner"
]

import os
import queue
import tempfile
import time
from multiprocessing import Process, Queue
from .conf import Conf, Configurable
from .log import zlog

class MyTaskRunnerConf(Conf):
    def __init__(self):
        self.num_worker = 0  # worker=0 means executing sync
        self.use_zmq = False  # whether using zmq?
        self.queue_size = 1000
        self.collector_timeout = 600.  # final collector timeout (seconds)

def get_zmq_socket(context, socket_type, endpoint: str, bind: bool, queue_size: int):
    import zmq
    buf_size = int(0.5 * 1024**3)  # hopefully this is enough
    socket = context.socket(socket_type)
    if socket_type == zmq.PUSH:
        socket.setsockopt(zmq.SNDHWM, queue_size)
        socket.setsockopt(zmq.SNDBUF, buf_size)
    elif socket_type == zmq.PULL:
        socket.setsockopt(zmq.RCVHWM, 0)
        socket.setsockopt(zmq.RCVBUF, buf_size)
    else:
        raise ValueError(f"Unsupported socket type: {socket_type}")
    if bind:
        socket.bind(endpoint)
    else:
        socket.connect(endpoint)
    return socket

# parallel task runner
# inputs -> [creator] -> tasks -> *[handler] -> results -> [collector]
@MyTaskRunnerConf.conf_rd()
class MyTaskRunner(Configurable):
    def __init__(self, conf: MyTaskRunnerConf = None, _task_create_func=None, _task_handle_func=None, **kwargs):
        super().__init__(conf, **kwargs)
        # --
        conf: MyTaskRunnerConf = self.conf
        self._task_create_func = _task_create_func  # allow simply passing them from the outside
        self._task_handle_func = _task_handle_func
        self._status = "idle"  # flag for checking
        # --
        self.processes = []
        self.sockets_filenames = None  # sockets for zmq
        self.queues = None  # queues for mp.Queue
        self._queue_size = conf.queue_size
        self._zmq_objects = {}
        if conf.num_worker > 0:
            if conf.use_zmq:  # use Process and zmq
                import zmq
                socketname_inputs, socketname_tasks, socketname_results = [tempfile.NamedTemporaryFile(delete=False).name for _ in range(3)]
                self.sockets_filenames = (socketname_inputs, socketname_tasks, socketname_results)
                _f1, _f2, _args = self._zmq_proc_create, self._zmq_proc_handle, self.sockets_filenames
                zlog(f"Create tmp files: {_args}")
                # --
                zmq_context = zmq.Context()
                zmq_inputs = get_zmq_socket(zmq_context, zmq.PUSH, f"ipc://{socketname_inputs}", True, conf.queue_size)
                zmq_results = get_zmq_socket(zmq_context, zmq.PULL, f"ipc://{socketname_results}", True, conf.queue_size)
                poller = zmq.Poller()
                poller.register(zmq_results, zmq.POLLIN)
                self._zmq_objects.update(zmq_context=zmq_context, zmq_inputs=zmq_inputs, zmq_results=zmq_results, poller=poller)
                # --
            else:
                queue_inputs, queue_tasks, queue_results = Queue(conf.queue_size), Queue(conf.queue_size), Queue(conf.queue_size)
                self.queues = (queue_inputs, queue_tasks, queue_results)
                _f1, _f2, _args = self._mp_proc_create, self._mp_proc_handle, self.queues
            # create processes
            p_creator = Process(target=_f1, args=_args[0:3])
            self.processes.append(p_creator)
            for _ in range(conf.num_worker):
                p_worker = Process(target=_f2, args=_args[1:3])
                self.processes.append(p_worker)
            for p in self.processes:
                p.start()
        # --

    def __del__(self):
        for p in self.processes:
            p.kill()
        if self.queues:
            for q in self.queues:
                q.close()
        if self.sockets_filenames:
            for f in self.sockets_filenames:
                if os.path.exists(f):
                    os.remove(f)
        if "zmq_context" in self._zmq_objects:
            self._zmq_objects["zmq_context"].destroy()

    # processes

    def _mp_proc_create(self, queue_inputs, queue_tasks, queue_results):
        while True:
            creator_inputs = queue_inputs.get()
            total_count = 0
            for task_data in self.task_create_func(creator_inputs):
                queue_tasks.put(task_data)
                total_count += 1
                #zlog(f"_mp_proc_create {task_data}")
            queue_results.put({"__ZZSignal__": True, "total_count": total_count})

    def _zmq_proc_create(self, socketname_inputs, socketname_tasks, socketname_results):
        import zmq
        context = zmq.Context()
        s_inputs = get_zmq_socket(context, zmq.PULL, f"ipc://{socketname_inputs}", False, self._queue_size)
        s_tasks = get_zmq_socket(context, zmq.PUSH, f"ipc://{socketname_tasks}", True, self._queue_size)
        s_results = get_zmq_socket(context, zmq.PUSH, f"ipc://{socketname_results}", False, self._queue_size)
        while True:
            creator_inputs = s_inputs.recv_json()
            total_count = 0
            for task_data in self.task_create_func(creator_inputs):
                s_tasks.send_json(task_data)
                total_count += 1
                #zlog(f"_zmq_proc_create ->{socketname_tasks} {task_data}")
            s_results.send_json({"__ZZSignal__": True, "total_count": total_count})

    def _mp_proc_handle(self, queue_tasks, queue_results):
        while True:
            task_data = queue_tasks.get()
            task_result = self.task_handle_func(task_data)
            queue_results.put(task_result)
            #zlog(f"_mp_proc_handle {task_result}")

    def _zmq_proc_handle(self, socketname_tasks, socketname_results):
        import zmq
        context = zmq.Context()
        s_tasks = get_zmq_socket(context, zmq.PULL, f"ipc://{socketname_tasks}", False, self._queue_size)
        s_results = get_zmq_socket(context, zmq.PUSH, f"ipc://{socketname_results}", False, self._queue_size)
        while True:
            task_data = s_tasks.recv_json()
            task_result = self.task_handle_func(task_data)
            s_results.send_json(task_result)
            #zlog(f"_zmq_proc_handle ->{socketname_results} {task_result}")

    # other helpers
    def _put_inputs(self, creator_inputs, queue_or_socket):
        conf: MyTaskRunnerConf = self.conf
        if conf.use_zmq:
            queue_or_socket.send_json(creator_inputs)
        else:
            queue_or_socket.put(creator_inputs)
        #zlog(f"_put_inputs {creator_inputs}")

    def _get_result(self, queue_or_socket, zmq_poller=None):
        conf: MyTaskRunnerConf = self.conf
        task_result = None
        if conf.use_zmq:
            # if zmq_poller.poll(conf.collector_timeout * 1000):
            #     task_result = queue_or_socket.recv_json()
            task_result = queue_or_socket.recv_json()
        else:
            try:
                task_result = queue_or_socket.get(timeout=conf.collector_timeout)
            except queue.Empty:
                pass
        #zlog(f"_get_result {task_result}")
        return task_result

    def yield_results(self, creator_inputs, max_result_count=0):
        assert self._status == "idle", "Cannot re-using a running TaskRunner"
        self._status = "working"
        conf: MyTaskRunnerConf = self.conf
        curr_count = 0
        if conf.num_worker <= 0:  # simply executing sync
            for task_data in self.task_create_func(creator_inputs):
                task_result = self.task_handle_func(task_data)
                curr_count += 1
                yield task_result
                if max_result_count > 0 and curr_count >= max_result_count:
                    break
        else:  # use extra processes
            if conf.use_zmq:
                s_inputs, s_results = self._zmq_objects["zmq_inputs"], self._zmq_objects["zmq_results"]
                poller = self._zmq_objects["poller"]
            else:
                s_inputs, _, s_results = self.queues
                poller = None
            self._put_inputs(creator_inputs, s_inputs)
            total_count = None
            while True:
                task_result = self._get_result(s_results, zmq_poller=poller)
                if task_result is None:
                    break  # break if timeout
                elif isinstance(task_result, dict) and "__ZZSignal__" in task_result:
                    total_count = task_result["total_count"]
                else:
                    curr_count += 1
                    yield task_result
                if total_count is not None and curr_count >= total_count:
                    break  # break if all gets collected
                if max_result_count > 0 and curr_count >= max_result_count:
                    break
        self._status = "idle"

    # --
    # to be implemented

    def task_create_func(self, inputs):
        if self._task_create_func:
            yield from self._task_create_func(inputs)
        else:
            raise NotImplementedError("To be implemented")

    def task_handle_func(self, data):
        if self._task_handle_func:
            return self._task_handle_func(data)
        else:
            raise NotImplementedError("To be implemented")

# --
def test_task_mp():
    TOTAL_COUNT = 100
    TIME_MAIN, TIME_DATA = 0.1, 0.05
    # --
    def _task_create_func(num):
        for idx in range(num["num"]):
            yield {"idx": idx}
    def _task_handle_func(data):
        ret = data.copy()
        time.sleep(TIME_DATA)
        ret["done"] = True
        return ret
    # --
    for use_zmq in [True, False]:
        for num_worker in [8, 4, 1, 0]:
            runner = MyTaskRunner(_task_create_func=_task_create_func, _task_handle_func=_task_handle_func, num_worker=num_worker, use_zmq=use_zmq)
            t1 = time.perf_counter()
            all_results = []
            for res in runner.yield_results({"num": TOTAL_COUNT}):
                time.sleep(TIME_MAIN)
                all_results.append(res)
            t2 = time.perf_counter()
            assert sorted([z['idx'] for z in all_results]) == list(range(TOTAL_COUNT))
            zlog(f"Run num_worker={num_worker} use_zmq={use_zmq} with {t2-t1} seconds, res_count={len(all_results)}")

# python -m mspx.utils.task_mp
if __name__ == '__main__':
    test_task_mp()
