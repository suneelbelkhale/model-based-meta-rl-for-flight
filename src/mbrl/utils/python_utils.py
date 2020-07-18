from collections import defaultdict
from dotmap import DotMap
import signal
import sys
import time


class AttrDict(DotMap):

    def __getitem__(self, item):
        if isinstance(item, str) and '/' in item:
            item_split = item.split('/')
            curr_item = item_split[0]
            next_item = '/'.join(item_split[1:])
            return self[curr_item][next_item]
        else:
            return super(AttrDict, self).__getitem__(item)

    def __setitem__(self, key, value):
        if isinstance(key, str) and '/' in key:
            key_split = key.split('/')
            curr_key = key_split[0]
            next_key = '/'.join(key_split[1:])
            self[curr_key][next_key] = value
        else:
            super(AttrDict, self).__setitem__(key, value)

    def pprint(self, str_max_len=5):
        str_self = self.leaf_apply(lambda x: str(x)[:str_max_len] + '...')
        return super(AttrDict, str_self).pprint(pformat='json')

    def leaf_keys(self):
        def _get_leaf_keys(d, prefix=''):
            for key, value in d.items():
                new_prefix = prefix + '/' + key if len(prefix) > 0 else key
                if isinstance(value, AttrDict):
                    yield from _get_leaf_keys(value, prefix=new_prefix)
                else:
                    yield new_prefix

        yield from _get_leaf_keys(self)

    def leaf_values(self):
        for key in self.leaf_keys():
            yield self[key]

    def leaf_items(self):
        for key in self.leaf_keys():
            yield key, self[key]

    def leaf_filter(self, func):
        d = AttrDict()
        for key, value in self.leaf_items():
            if func(key, value):
                d[key] = value
        return d

    def leaf_assert(self, func):
        """
        Recursively asserts func on each value
        :param func (lambda): takes in one argument, outputs True/False
        """
        for value in self.leaf_values():
            assert func(value)

    def leaf_modify(self, func):
        """
        Applies func to each value (recursively), modifying in-place
        :param func (lambda): takes in one argument and returns one object
        """
        for key, value in self.leaf_items():
            self[key] = func(value)

    def leaf_kv_modify(self, func):
        """
        Applies func to each value (recursively), modifying in-place
        :param func (lambda): takes in two arguments and returns one object
        """
        for key, value in self.leaf_items():
            self[key] = func(key, value)

    def leaf_apply(self, func):
        """
        Applies func to each value (recursively) and returns a new AttrDict
        :param func (lambda): takes in one argument and returns one object
        :return AttrDict
        """
        d_copy = self.copy()
        d_copy.leaf_modify(func)
        return d_copy

    def combine(self, d_other):
        for k, v in d_other.leaf_items():
            self[k] = v

    def freeze(self):
        frozen = AttrDict(self, _dynamic=False)
        self.__dict__.update(frozen.__dict__)
        return self

    @staticmethod
    def leaf_combine_and_apply(ds, func, map_func=lambda x: x, match_keys=True):
        leaf_keys = tuple(sorted(ds[0].leaf_keys()))
        if match_keys:
            for d in ds[1:]:
                assert leaf_keys == tuple(sorted(d.leaf_keys()))

        d_combined = AttrDict()
        for k in leaf_keys:
            values = [map_func(d[k]) for d in ds]
            d_combined[k] = func(values)

        return d_combined


    @staticmethod
    def from_dict(d):
        d_attr = AttrDict()
        for k, v in d.items():
            d_attr[k] = v
        return d_attr

class TimeIt(object):
    def __init__(self, prefix=''):
        self.prefix = prefix
        self.start_times = dict()
        self.elapsed_times = defaultdict(int)

        self._with_name_stack = []

        self._with_args_stack = []

    def __call__(self, name, reset_on_stop=False):
        self._with_name_stack.append(name)
        self._with_args_stack.append({"reset_on_stop": reset_on_stop})
        return self

    def __enter__(self):
        self.start(self._with_name_stack[-1], **self._with_args_stack[-1])
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        name = self._with_name_stack.pop()
        kwargs = self._with_args_stack.pop()
        timeit.stop(name, **kwargs)

    def start(self, name, **kwargs):
        assert(name not in self.start_times)
        self.start_times[name] = time.time()

    def stop(self, name, reset_on_stop=False, **kwargs):
        assert(name in self.start_times)
        if reset_on_stop:
            self.elapsed_times[name] = 0
        self.elapsed_times[name] += time.time() - self.start_times[name]
        self.start_times.pop(name)

    def elapsed(self, name):
        return self.elapsed_times[name]

    def reset(self):
        self.start_times = dict()
        self.elapsed_times = defaultdict(int)

    def __str__(self):
        s = ''
        names_elapsed = sorted(self.elapsed_times.items(), key=lambda x: x[1], reverse=True)
        for name, elapsed in names_elapsed:
            if 'total' not in self.elapsed_times:
                s += '{0}: {1: <10} {2:.1f}\n'.format(self.prefix, name, elapsed)
            else:
                assert(self.elapsed_times['total'] >= max(self.elapsed_times.values()))
                pct = 100. * elapsed / self.elapsed_times['total']
                s += '{0}: {1: <10} {2:.1f} ({3:.1f}%)\n'.format(self.prefix, name, elapsed, pct)
        if 'total' in self.elapsed_times:
            times_summed = sum([t for k, t in self.elapsed_times.items() if k != 'total'])
            other_time = self.elapsed_times['total'] - times_summed
            assert(other_time >= 0)
            pct = 100. * other_time / self.elapsed_times['total']
            s += '{0}: {1: <10} {2:.1f} ({3:.1f}%)\n'.format(self.prefix, 'other', other_time, pct)
        return s

timeit = TimeIt()


def exit_on_ctrl_c():
    def signal_handler(signal, frame):
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)


if __name__ == '__main__':
    d = AttrDict(dict(
        a=dict(
            b=1,
            c=2
        )
    ))
    # print('start')
    # print(d['a/b'])
    # print(d['a/c'])
    # d['a/d'] = 3
    # print(d['a/d'])
    print(d.pprint())

    d1 = AttrDict(
        a=AttrDict(
            e=4
        )
    )
    d.combine(d1)

    d.pprint()
