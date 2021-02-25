from fnmatch import fnmatch


class GlobbingFilter:
    '''
    Filter module names using a set of globs.

    Objects are matched against the exclude list first, then the include list.
    Anything that passes through without matching either, is excluded.
    '''
    # 先用include选中几个大package， 然后再用exclude排除这几个package中一些不想要的
    # 排除分为两种跳过和终止后续
    # 目前的逻辑是终止后续， 先把这类做好吧。  比如一个类下面很多方法，到这个类就可以了，不用深入
    def __init__(self, include=None, exclude=None):
        if include is None and exclude is None:
            include = ['*']
            exclude = []
        elif include is None:
            include = ['*']
        elif exclude is None:
            exclude = []

        self.include = include
        self.exclude = exclude

    def __call__(self, full_name=None):
        for pattern in self.exclude:
            if fnmatch(full_name, pattern):
                return False

        for pattern in self.include:
            if fnmatch(full_name, pattern):
                return True

        return False
