import pickle


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def write_pickle(x, path):
    with open(path, 'wb') as f:
        pickle.dump(x, f)

'''返回所有的tag类型'''
def load_lines(path, encoding='utf8'):
    with open(path, 'r', encoding=encoding) as f:
        lines = [line.strip() for line in f.readlines()]#line='B-Medical_Examination\n'  去掉换行
        return lines


def write_lines(lines, path, encoding='utf8'):
    with open(path, 'w', encoding=encoding) as f:
        for line in lines:
            f.writelines('{}\n'.format(line))


'''
模块 pickle 实现了对一个 Python 对象结构的二进制序列化和反序列化。 "pickling" 是将 Python 对象及其所拥有的层次结构转化为一个字节流的过程，而 "unpickling" 是相反的操作，
会将（来自一个 binary file 或者 bytes-like object 的）字节流转化回一个对象层次结构。 pickling（和 unpickling）也被称为“序列化”, “编组” 1 或者 “平面化”。而为了避免混乱，
此处采用术语 “封存 (pickling)” 和 “解封 (unpickling)”。

默认情况下，JSON 只能表示 Python 内置类型的子集，不能表示自定义的类；但 pickle 可以表示大量的 Python 数据类型（可以合理使用 Python 的对象内省功能自动地表示大多数类型，
复杂情况可以通过实现 specific object APIs 来解决）。
'''