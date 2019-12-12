from __future__ import division

import inspect
import sys
import time
from distutils import sysconfig
from collections import defaultdict
from threading import Thread
from queue import Queue, Empty
from .util import Util
import os


# try:
#     from Queue import Queue, Empty
# except ImportError:
#     from queue import Queue, Empty  # py3中只有这个


# from beeprint import pp,Config
# bp_config = Config()
# fo  = open("/Users/zhangxingjun/pycode/pycallgraph/p.log", "w")

# bp_config.stream = fo

class SyncronousTracer(object):

    def __init__(self, outputs, config):
        self.processor = TraceProcessor(outputs, config)
        self.config = config

    def tracer(self, frame, event, arg):
        self.processor.process(frame, event, arg, self.memory())
        return self.tracer

    def memory(self):
        if self.config.memory:
            from .memory_profiler import memory_usage
            return int(memory_usage(-1, 0)[0] * 1000000)

    def start(self):
        sys.settrace(self.tracer)

    def stop(self):
        sys.settrace(None)

    def done(self):
        pass


class AsyncronousTracer(SyncronousTracer):

    def start(self):
        self.processor.start()
        SyncronousTracer.start(self)

    def tracer(self, frame, event, arg):
        self.processor.queue(frame, event, arg, self.memory())
        return self.tracer

    def done(self):
        self.processor.done()
        self.processor.join()


class TraceProcessor(Thread):
    '''
    Contains a callback used by sys.settrace, which collects information about
    function call count, time taken, etc.
    '''

    def __init__(self, outputs, config):
        Thread.__init__(self)
        self.trace_queue = Queue()
        self.keep_going = True
        self.outputs = outputs
        self.config = config
        self.updatables = [a for a in self.outputs if a.should_update()]

        self.init_trace_data()
        self.init_libpath()
        # self.log = open("/Users/zhangxingjun/pycode/pycallgraph/p.log", "w")

    def init_trace_data(self):
        self.previous_event_return = False

        # A mapping of which function called which other function
        self.call_dict = defaultdict(lambda: defaultdict(int))

        # Current call stack
        self.call_stack = ['__main__']

        # Counters for each function
        self.func_count = defaultdict(int)
        self.func_count_max = 0
        self.func_count['__main__'] = 1

        # Accumulative time per function
        self.func_time = defaultdict(float)
        self.func_time_max = 0

        # Accumulative memory addition per function
        self.func_memory_in = defaultdict(int)
        self.func_memory_in_max = 0

        # Accumulative memory addition per function once exited
        self.func_memory_out = defaultdict(int)
        self.func_memory_out_max = 0

        # Keeps track of the start time of each call on the stack
        self.call_stack_timer = []
        self.call_stack_memory_in = []
        self.call_stack_memory_out = []

    def init_libpath(self):
        self.lib_path = sysconfig.get_python_lib()
        # zhangxingjun
        # path = os.path.split(self.lib_path)
        # if path[1] == 'site-packages':
        #     self.lib_path = path[0]
        # self.lib_path = self.lib_path.lower()

    def queue(self, frame, event, arg, memory):
        data = {
            'frame': frame,
            'event': event,
            'arg': arg,
            'memory': memory,
        }
        self.trace_queue.put(data)

    def run(self):
        while self.keep_going:
            try:
                data = self.trace_queue.get(timeout=0.1)
            except Empty:
                pass
            self.process(**data)

    def done(self):
        while not self.trace_queue.empty():
            time.sleep(0.1)
        self.keep_going = False
        # self.log.close()

    def process(self, frame, event, arg, memory=None):
        '''This function processes a trace result. Keeps track of relationships between calls.

        frame有以下属性
        f_back	前一个堆栈帧（朝向调用者），如果这是底部堆栈帧则为None
        f_code	在这个框架中执行的Code对象
        f_locals	用于查找局部变量的字典
        f_globals	用于全局变量
        f_builtins	用于内置名称
        f_restricted	表示该函数是否在限制执行模式下执行的标志
        f_lasti	给出精确的指令（这是代码对象的字节码字符串的索引）

        然后通过 inspect.getmodule(code) 获取当前module class  func 详细信息，还可以获取堆栈信息

        event
        'call' 调用一个函数（或输入一些其他代码块）。调用全局跟踪函数; arg是None; 返回值指定本地跟踪功能。
        'line' 解释器即将执行新的代码行或重新执行循环的条件。调用本地跟踪功能; arg是 None; 返回值指定新的本地跟踪功能。有关Objects/lnotab_notes.txt其工作原理的详细说明，请参阅 。
        'return' 函数（或其他代码块）即将返回。调用本地跟踪功能; arg是将返回的值，或者None 事件是由引发的异常引起的。跟踪函数的返回值被忽略。
        'exception' 发生了一个例外。调用本地跟踪功能; arg是一个元组; 返回值指定新的本地跟踪功能。(exception, value, traceback)
        '''

        if memory is not None and self.previous_event_return:
            # Deal with memory when function has finished so local variables can be cleaned up
            self.previous_event_return = False
            if self.call_stack_memory_out:
                full_name, m = self.call_stack_memory_out.pop(-1)
            else:
                full_name, m = (None, None)
            # NOTE: Call stack is no longer the call stack that may be expected. Potentially need to store a copy of it.
            if full_name and m:
                call_memory = memory - m
                self.func_memory_out[full_name] += call_memory
                self.func_memory_out_max = max(self.func_memory_out_max, self.func_memory_out[full_name])

        if event == 'call':
            keep = True
            isStdlib = False
            code = frame.f_code

            # Stores all the parts of a human readable name of the current call
            full_name_list = []
            module = inspect.getmodule(code)  # Work out the module name
            # pp(module,max_depth=2,config=bp_config)
            if module:
                module_name = module.__name__
                module_path = module.__file__
                if not self.config.include_stdlib and self.is_module_stdlib(module_path):
                    keep = False
                    isStdlib = True
                if module_name == '__main__':
                    module_name = ''
                # print(module_name, "\t", module_path, "\t", str(keep), file=self.log)  # 张行军
            else:
                module_name = ''

            if module_name:
                full_name_list.append(module_name)

            # Work out the class name
            try:
                class_name = frame.f_locals['self'].__class__.__name__
                full_name_list.append(class_name)
            except (KeyError, AttributeError):
                class_name = ''

            # Work out the current function or method
            func_name = code.co_name
            if func_name == '?':
                func_name = '__main__'
            full_name_list.append(func_name)
            full_name = '.'.join(full_name_list)  # Create a readable representation of the current call
            # print(module_name, "\t", class_name, "\t", full_name, file=self.log)  # 张行军

            if len(self.call_stack) > self.config.max_depth:
                keep = False
            if keep and self.config.trace_filter:
                keep = self.config.trace_filter(full_name)

            # Store the call information
            if keep:
                if self.call_stack:
                    # src_func = self.call_stack[-1]  # 从堆栈中获取调用者
                    # 张行军 20191126 ，如果不保持的 记为''， 则后者从前面获取父节点，相当于在图中跳过该点，将前后节点连起来
                    # 对于keep==flase 的，可以进一步区分控制，比如遇到想终止后续的，而不是跳过的 ,这种self.call_stack.append(None)
                    src_func = None
                    i = len(self.call_stack)
                    while i > 0:
                        i = i - 1
                        if self.call_stack[i] is None:  # 先判断有没有要终止绘制后续的
                            src_func = ''
                            break
                        if self.call_stack[i] != '':
                            src_func = self.call_stack[i]
                            break
                else:
                    src_func = None
                self.call_dict[src_func][full_name] += 1
                self.func_count[full_name] += 1
                self.func_count_max = max(self.func_count_max, self.func_count[full_name])
                self.call_stack.append(full_name)
                # print("----".join(self.call_stack), file=fo)  # 张行军
                self.call_stack_timer.append(time.time())
                if memory is not None:
                    self.call_stack_memory_in.append(memory)
                    self.call_stack_memory_out.append([full_name, memory])
            else:
                if isStdlib:
                    self.call_stack.append(None)
                else:
                    self.call_stack.append('')
                self.call_stack.append('')  # 要想更丰富一点，应该扩展该list中item类型， 比如里面是个dict，
                self.call_stack_timer.append(None)

        if event == 'return':
            self.previous_event_return = True

            if self.call_stack:
                full_name = self.call_stack.pop(-1)

                if self.call_stack_timer:
                    start_time = self.call_stack_timer.pop(-1)
                else:
                    start_time = None

                if start_time:
                    call_time = time.time() - start_time
                    self.func_time[full_name] += call_time
                    self.func_time_max = max(self.func_time_max, self.func_time[full_name])

                if memory is not None:
                    if self.call_stack_memory_in:
                        start_mem = self.call_stack_memory_in.pop(-1)
                    else:
                        start_mem = None
                    if start_mem:
                        call_memory = memory - start_mem
                        self.func_memory_in[full_name] += call_memory
                        self.func_memory_in_max = max(self.func_memory_in_max, self.func_memory_in[full_name], )

    def is_module_stdlib(self, file_name):
        return not file_name.lower().startswith(self.lib_path)  # 张行军 self.lib_path 为 site-packages ,所以加了 not

    def __getstate__(self):
        '''
        Used for when creating a pickle. Certain instance variables can't pickled and aren't used anyway.
        '''
        odict = self.__dict__.copy()
        dont_keep = [
            'outputs',
            'config',
            'updatables',
            'lib_path',
        ]
        for key in dont_keep:
            del odict[key]

        return odict

    def groups(self):
        grp = defaultdict(list)
        for node in self.nodes():
            grp[node.group].append(node)
        for g in grp.items():  # .iteritems():
            yield g

    def stat_group_from_func(self, func, calls):
        stat_group = StatGroup()
        stat_group.name = func
        stat_group.group = self.config.trace_grouper(func)
        stat_group.calls = Stat(calls, self.func_count_max)
        stat_group.time = Stat(self.func_time.get(func, 0), self.func_time_max)
        stat_group.memory_in = Stat(self.func_memory_in.get(func, 0), self.func_memory_in_max)
        stat_group.memory_out = Stat(self.func_memory_in.get(func, 0), self.func_memory_in_max)
        return stat_group

    def nodes(self):
        for func, calls in self.func_count.items():  # .iteritems():
            yield self.stat_group_from_func(func, calls)

    def edges(self):
        # 不keep应该有两种，一是该点之后的都不保留，二是跳过该点，所以可以在参数config.trace_filter.exclude中
        for src_func, dests in self.call_dict.items():  # .iteritems():
            if src_func:
                for dst_func, calls in dests.items():  # .iteritems():
                    edge = self.stat_group_from_func(dst_func, calls)
                    edge.src_func = src_func
                    edge.dst_func = dst_func
                    yield edge


class Stat(object):
    '''
    Stores a "statistic" value, e.g. "time taken" along with the maximum
    possible value of the value, which is used to calculate the fraction of 1.
    The fraction is used for choosing colors.
    '''

    def __init__(self, value, total):
        self.value = value
        self.total = total
        try:
            self.fraction = value / total
        except ZeroDivisionError:
            self.fraction = 0

    @property
    def value_human_bibyte(self):
        '''Mebibyte of the value in human readable a form.'''
        return Util.human_readable_bibyte(self.value)


class StatGroup(object):
    pass


def simple_memoize(callable_object):
    '''
    Simple memoization for functions without keyword arguments.

    This is useful for mapping code objects to module in this context.
    inspect.getmodule() requires a number of system calls, which may slow down
    the tracing considerably. Caching the mapping from code objects (there is
    *one* code object for each function, regardless of how many simultaneous
    activations records there are).

    In this context we can ignore keyword arguments, but a generic memoizer
    ought to take care of that as well.
    '''

    cache = dict()

    def wrapper(*rest):
        if rest not in cache:
            cache[rest] = callable_object(*rest)
        return cache[rest]

    return wrapper


inspect.getmodule = simple_memoize(inspect.getmodule)
