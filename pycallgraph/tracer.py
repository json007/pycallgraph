import inspect
import sys
import time
from collections import defaultdict
from threading import Thread
from queue import Queue, Empty
from .util import Util
from fnmatch import fnmatch
import dill

class SyncronousTracer:

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

    def save(self):
        self.processor.save()

    def load(self):
        self.processor.load()

    def save_func_name(self):
        self.processor.save_func_name()

    def prune(self):
        self.processor.prune()


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

    def prune(self):
        self.processor.prune()


class TraceProcessor(Thread):
    '''
    Contains a callback used by sys.settrace, which collects information about function call count, time taken, etc.
    '''

    def __init__(self, outputs, config):
        Thread.__init__(self)
        self.trace_queue = Queue()
        self.keep_going = True
        self.config = config

        self.outputs = outputs
        self.updatables = [a for a in self.outputs if a.should_update()] # 这个类中，updatables和outputs 都没用上

        self.init_trace_data()

        today = time.strftime("%Y%m%d", time.localtime())
        # self.log = open(f"/mnt/d/code/pycallgraph/{today}.log", "w")
        # self.log = open(f"D:/code/pycallgraph/{today}.log", "w")
        # self.package_prefix = package_prefix

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

    def is_module_stdlib(self, file_name):
        return file_name.lower().find('site-packages')<0  # 张行军 找不到site-packages则是标准库

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
                self.process(**data)
            except Empty:
                pass

    def done(self):
        while not self.trace_queue.empty():
            time.sleep(0.1)
        self.keep_going = False
        # self.log.close()

    def process(self, frame, event, arg, memory=None):
        '''This function processes a trace result. Keeps track of relationships between calls.

        frame有以下属性：
            f_back	前一个堆栈帧（朝向调用者），如果这是底部堆栈帧则为None
            f_code	在这个框架中执行的Code对象
            f_locals	用于查找局部变量的字典
            f_globals	用于全局变量
            f_builtins	用于内置名称
            f_restricted	表示该函数是否在限制执行模式下执行的标志
            f_lasti	给出精确的指令（这是代码对象的字节码字符串的索引）
        然后通过 inspect.getmodule(frame.f_code) 获取当前module class  func 详细信息，还可以获取堆栈信息

        event有四种：
        'call' 调用一个函数（或输入一些其他代码块）。调用全局跟踪函数; arg是None; 返回值指定本地跟踪功能。
        'line' 解释器即将执行新的代码行或重新执行循环的条件。调用本地跟踪功能; arg是 None; 返回值指定新的本地跟踪功能。有关Objects/lnotab_notes.txt其工作原理的详细说明，请参阅 。
        'return' 函数（或其他代码块）即将返回。调用本地跟踪功能; arg是将返回的值，或者None 事件是由引发的异常引起的。跟踪函数的返回值被忽略。
        'exception' 发生了一个例外。调用本地跟踪功能; arg是一个元组; 返回值指定新的本地跟踪功能。(exception, value, traceback)

        python的类名返回太糟糕了， 此副本只处理某个包中的函数。
        '''

        if memory is not None and self.previous_event_return:
            # Deal with memory when function has finished so local variables can be cleaned up
            self.previous_event_return = False
            if self.call_stack_memory_out:
                full_name, m = self.call_stack_memory_out.pop(-1)
            else:
                full_name, m = (None, None)
            if full_name and m:
                self.func_memory_out[full_name] +=  memory - m
                self.func_memory_out_max = max(self.func_memory_out_max, self.func_memory_out[full_name])

        if event == 'call':
            full_name_list = [] # Stores all the parts of a human readable name of the current call
            # Work out the module name 重点
            module = inspect.getmodule(frame.f_code)  
            if module:
                # pp(module,max_depth=2,config=bp_config)
                module_name = module.__name__
                if module_name != '__main__':
                    full_name_list.append(module_name)
            # Work out the class name 类名
            try:
                full_name_list.append(frame.f_locals['self'].__class__.__name__)
            except (KeyError, AttributeError):
                pass
            # Work out the current function or method 函数或方法名
            func_name = frame.f_code.co_name
            if func_name == '?':
                func_name = '__main__'
            full_name_list.append(func_name)
            full_name = '.'.join(full_name_list)  # Create a readable representation of the current call
            # print(full_name, file=self.log)  # 张行军

            keep = True
            if len(self.call_stack) > self.config.max_depth:
                keep = False
            # if keep and self.config.trace_filter:
            #     keep = self.config.trace_filter(full_name)

            # Store the call information
            if keep:
                if self.call_stack:
                    src_func = self.call_stack[-1]  # 从堆栈中获取调用者
                    # # 张行军 20191126，如果不保持的，记为''， 则后者从前面获取父节点，相当于在图中跳过该点，将前后节点连起来
                    # # 对于想终止后续的，而不是跳过的, 这种self.call_stack.append(None)
                    # # 最后绘图时都可以通过 keep==flase 判断
                    # src_func = None
                    # i = len(self.call_stack)
                    # while i > 0:
                    #     i = i - 1
                    #     if self.call_stack[i] is None:  # 先判断有没有要终止绘制后续的
                    #         src_func = ''
                    #         break
                    #     if self.call_stack[i] != '':
                    #         src_func = self.call_stack[i]
                    #         break
                else:
                    src_func = None

                self.call_dict[src_func][full_name] += 1
                self.func_count[full_name] += 1
                self.func_count_max = max(self.func_count_max, self.func_count[full_name])
                self.call_stack.append(full_name)
                self.call_stack_timer.append(time.time())
                # print("----".join(self.call_stack), file=fo)  # 张行军

                if memory is not None:
                    self.call_stack_memory_in.append(memory)
                    self.call_stack_memory_out.append([full_name, memory])
            else:
                self.call_stack.append('')  # 要想更丰富一点，应该扩展该list中item类型， 比如里面是个dict，
                self.call_stack_timer.append(None)

        if event == 'return':
            self.previous_event_return = True
            if self.call_stack:
                full_name = self.call_stack.pop() # pop(-1) 默认的index就是-1
                if self.call_stack_timer:
                    start_time = self.call_stack_timer.pop()
                else:
                    start_time = None
                if start_time:
                    self.func_time[full_name] += time.time() - start_time
                    self.func_time_max = max(self.func_time_max, self.func_time[full_name])

                if memory is not None:
                    if self.call_stack_memory_in:
                        start_mem = self.call_stack_memory_in.pop(-1)
                    else:
                        start_mem = None
                    if start_mem:
                        self.func_memory_in[full_name] += memory - start_mem
                        self.func_memory_in_max = max(self.func_memory_in_max, self.func_memory_in[full_name])


#################### 下面都是在output.graphviz中调用的 #############################

    def groups(self):
        grp = defaultdict(list)
        for node in self.nodes():
            grp[node.group].append(node)
        for g in grp.items():
            yield g

    # def nodes(self):
    #     for func, calls in self.func_count.items():
    #         if self.config.trace_filter(func):# zhangxingjun 
    #             func_new = self.drop_prefix(func) # zhangxingjun 
    #             yield self.stat_group_from_func(func, func_new,calls)

    # def edges(self):
    #     # 不keep应该有两种，一是该点之后的都不保留，二是跳过该点，所以可以在参数config.trace_filter.exclude中
    #     for src_func, dests in self.call_dict.items():
    #         if src_func and self.config.trace_filter(src_func):
    #             src_func_new = self.drop_prefix(src_func) # zhangxingjun 
    #             for dst_func, calls in dests.items():
    #                 if self.config.trace_filter(dst_func):
    #                     dst_func_new = self.drop_prefix(dst_func) # zhangxingjun 
    #                     edge = self.stat_group_from_func(dst_func, dst_func_new, calls)
    #                     edge.src_func = src_func_new
    #                     edge.dst_func = dst_func_new
    #                     yield edge

    
    # def stat_group_from_func(self, func, func_new, calls):
    #     stat_group = StatGroup()
    #     stat_group.name = func_new
    #     stat_group.group = self.config.trace_grouper(func_new)
    #     stat_group.calls = Stat(calls, self.func_count_max)
    #     stat_group.time = Stat(self.func_time.get(func, 0), self.func_time_max)
    #     stat_group.memory_in = Stat(self.func_memory_in.get(func, 0), self.func_memory_in_max)
    #     stat_group.memory_out = Stat(self.func_memory_in.get(func, 0), self.func_memory_in_max)
    #     return stat_group

    def nodes(self):
        for func, calls in self.func_count.items():
            yield self.stat_group_from_func(func,calls)

    def edges(self):
        # 不keep应该有两种，一是该点之后的都不保留，二是跳过该点，所以可以在参数config.trace_filter.exclude中
        for src_func, dests in self.call_dict.items():
            for dst_func, calls in dests.items():
                edge = self.stat_group_from_func(dst_func, calls)
                edge.src_func = src_func
                edge.dst_func = dst_func
                yield edge

    
    def stat_group_from_func(self, func, calls):
        stat_group = StatGroup()
        stat_group.name = func
        stat_group.group = self.config.trace_grouper(func)
        stat_group.calls = Stat(calls, self.func_count_max)
        stat_group.time = Stat(self.func_time.get(func, 0), self.func_time_max)
        stat_group.memory_in = Stat(self.func_memory_in.get(func, 0), self.func_memory_in_max)
        stat_group.memory_out = Stat(self.func_memory_in.get(func, 0), self.func_memory_in_max)
        return stat_group



    def drop_prefix(self, func):
        l = len(self.config.package_prefix)
        if func[0:l] ==  self.config.package_prefix: # 'pytorch_lightning.':
            func = func[l:]
        for s,c in self.config.func_name_prune.items():
            if fnmatch(func, s):
                t = func.split('.')[:-c]
                return '.'.join(t)
        return func

    def prune(self): # 对结果进行过滤 和 节点名字”掐头去尾“，  以及合并相同的节点
        new_call_dict = defaultdict(lambda: defaultdict(int))
        new_func_count = defaultdict(int)
        new_func_time = defaultdict(float)
        new_func_memory_in = defaultdict(int)

        for func, calls in self.func_count.items():
            if self.config.trace_filter(func):# zhangxingjun 
                new_func = self.drop_prefix(func)
                new_func_count[new_func] = new_func_count[new_func] + calls
                new_func_time[new_func] = new_func_time[new_func] + self.func_time.get(func, 0)
                new_func_memory_in[new_func] = new_func_memory_in[new_func] + self.func_memory_in.get(func, 0)

        for src_func, dests in self.call_dict.items():
            if src_func and self.config.trace_filter(src_func):
                src_func_new = self.drop_prefix(src_func) # zhangxingjun 
                for dst_func, calls in dests.items():
                    if self.config.trace_filter(dst_func):
                        dst_func_new = self.drop_prefix(dst_func) # zhangxingjun 
                        new_call_dict[src_func_new][dst_func_new] = new_call_dict[src_func_new][dst_func_new] + calls
        self.func_count = new_func_count
        self.func_time = new_func_time
        self.func_memory_in = new_func_memory_in
        self.call_dict = new_call_dict


    def save_func_name(self):
        fh = open(self.config.full_func_name_file, 'w')
        for func ,_ in self.func_count.items():
            print(func, file=fh)
        fh.close()

    def save(self):
        d = {
            'func_count':self.func_count,
            'func_time':self.func_time,
            'func_memory_in':self.func_memory_in,
            'call_dict':self.call_dict,
        }
        dill.dump(d, open(self.config.tracker_log,"wb")) # 张行军

    def load(self):
        d = dill.load(open(self.config.tracker_log,"rb"))
        self.func_count = d['func_count']
        self.func_time = d['func_time']
        self.func_memory_in = d['func_memory_in']
        self.call_dict = d['call_dict']


class Stat:
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


class StatGroup:
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
