# 高级编程

## 91. 元类（Metaclass）的作用与实现
问题：什么是元类？编写一个元类示例，实现类的单例模式（Singleton）。

答案：
```python
class SingletonMeta(type):
    _instances = {}
    def __call__(cls, *args, kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, kwargs)
        return cls._instances[cls]

class SingletonClass(metaclass=SingletonMeta):
    def __init__(self, value):
        self.value = value

# 测试单例
obj1 = SingletonClass(10)
obj2 = SingletonClass(20)
print(obj1.value, obj2.value)  # 输出: 20 20（两个对象是同一个实例）
```

解释：
- 元类是类的类，控制类的创建行为。
- `__call__` 方法在创建类实例时被调用，确保每个类只有一个实例。



## 92. 上下文管理器与 `__enter__`/`__exit__`
问题：如何通过类实现上下文管理器？编写一个支持 `with` 语句的文件读取类。

答案：
```python
class FileReader:
    def __init__(self, filename):
        self.filename = filename
    
    def __enter__(self):
        self.file = open(self.filename, 'r')
        return self.file
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()
        if exc_type:
            print(f"Error occurred: {exc_val}")
        return True  # 抑制异常传播

# 使用示例
with FileReader("example.txt") as f:
    content = f.read()
    print(content)
```

解释：
- `__enter__` 返回资源对象（如文件句柄）。
- `__exit__` 处理清理工作，并可选处理异常。



## 93. 生成器（Generator）与协程（Coroutine）的区别
问题：解释生成器和协程的区别，并编写一个生成器和一个协程的示例。

答案：
```python
# 生成器：生成数据（使用 yield）
def simple_generator(n):
    for i in range(n):
        yield i * 2

gen = simple_generator(3)
print(list(gen))  # 输出 [0, 2, 4]

# 协程：消费数据（使用 yield 接收值）
def coroutine():
    while True:
        x = yield
        print(f"Received: {x}")

co = coroutine()
next(co)  # 启动协程
co.send(10)  # 输出 "Received: 10"
```

解释：
- 生成器用 `yield` 生成值，协程用 `yield` 接收值。
- 协程需要先调用 `next()` 启动，生成器可直接迭代。



## 94. 使用 `asyncio` 实现异步任务
问题：编写一个异步函数，并发执行两个HTTP请求（模拟），并返回结果。

答案：
```python
import asyncio

async def mock_http_request(url):
    await asyncio.sleep(1)  # 模拟IO等待
    return f"Response from {url}"

async def main():
    # 并发执行两个任务
    task1 = asyncio.create_task(mock_http_request("url1"))
    task2 = asyncio.create_task(mock_http_request("url2"))
    responses = await asyncio.gather(task1, task2)
    print(responses)  # 输出两个响应结果

asyncio.run(main())
```

解释：
- `asyncio.create_task` 创建并发任务。
- `await asyncio.gather` 等待所有任务完成。



## 95. 垃圾回收与循环引用
问题：Python如何解决循环引用问题？编写一个产生循环引用的示例，并手动触发垃圾回收。

答案：
```python
import gc

class Node:
    def __init__(self):
        self.child = None

# 创建循环引用
a = Node()
b = Node()
a.child = b
b.child = a

# 删除引用
del a, b

# 手动触发垃圾回收（分代回收）
gc.collect()
print("循环引用已清理") if gc.garbage == [] else print("存在未清理的循环引用")
```

解释：
- Python的垃圾回收器使用分代回收和标记清除检测循环引用。
- `gc.collect()` 强制触发回收，`gc.garbage` 显示无法回收的对象。



## 96. 多线程与GIL（全局解释器锁）
问题：解释GIL对多线程的影响，并编写一个多线程计算密集型任务的示例。

答案：
```python
import threading
import time

def count(n):
    while n > 0:
        n -= 1

# 单线程执行
start = time.time()
count(100_000_000)
print(f"单线程耗时: {time.time() - start:.2f}秒")

# 多线程执行（由于GIL，可能更慢）
t1 = threading.Thread(target=count, args=(50_000_000,))
t2 = threading.Thread(target=count, args=(50_000_000,))
start = time.time()
t1.start()
t2.start()
t1.join()
t2.join()
print(f"双线程耗时: {time.time() - start:.2f}秒")
```

解释：
- GIL确保同一时刻只有一个线程执行字节码，CPU密集型任务多线程反而更慢。
- IO密集型任务（如网络请求）多线程仍有优势。



## 97. 魔术方法 `__slots__` 的优化作用
问题：`__slots__` 如何优化内存？编写一个对比类，展示其内存占用差异。

答案：
```python
class WithoutSlots:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class WithSlots:
    __slots__ = ('x', 'y')
    def __init__(self, x, y):
        self.x = x
        self.y = y

# 内存占用测试
from sys import getsizeof

obj1 = WithoutSlots(1, 2)
obj2 = WithSlots(1, 2)
print(getsizeof(obj1))  # 典型值：56（更高）
print(getsizeof(obj2))  # 典型值：48（更低）
```

解释：
- `__slots__` 通过固定属性列表，避免每个实例使用 `__dict__` 动态存储，节省内存。
- 适用于需要创建大量实例的场景。



## 98. 动态修改类与猴子补丁（Monkey Patching）
问题：如何动态修改类的方法？编写一个示例，为现有类添加新方法。

答案：
```python
class OriginalClass:
    def original_method(self):
        return "Original"

# 动态添加方法
def new_method(self):
    return "Patched"

OriginalClass.new_method = new_method

obj = OriginalClass()
print(obj.new_method())  # 输出 "Patched"
```

解释：
- 猴子补丁在运行时修改类或模块的行为。
- 常用于临时修复第三方库的问题，但可能引入维护风险。



## 99. 描述符（Descriptor）的应用
问题：用描述符实现一个类型检查属性，确保属性值为整数。

答案：
```python
class IntegerField:
    def __set_name__(self, owner, name):
        self.name = name
    
    def __get__(self, instance, owner):
        return instance.__dict__.get(self.name)
    
    def __set__(self, instance, value):
        if not isinstance(value, int):
            raise TypeError("Value must be an integer")
        instance.__dict__[self.name] = value

class Person:
    age = IntegerField()

p = Person()
p.age = 30  # 正确
p.age = "30"  # 抛出 TypeError
```

解释：
- 描述符通过 `__get__` 和 `__set__` 控制属性访问。
- 常用于ORM框架（如Django模型字段）。



## 100. 使用 `ctypes` 调用C函数
问题：如何通过 `ctypes` 调用C标准库的 `printf` 函数？编写示例代码。

答案：
```python
from ctypes import cdll, c_char_p

# 加载C标准库
libc = cdll.LoadLibrary("libc.so.6")  # Linux
# libc = cdll.msvcrt  # Windows

# 定义函数参数和返回类型
libc.printf.argtypes = [c_char_p]
libc.printf.restype = None

# 调用printf
libc.printf(b"Hello from C!\n")  # 输出 "Hello from C!"
```

解释：
- `ctypes` 允许Python调用动态链接库中的C函数。
- `argtypes` 和 `restype` 指定参数和返回类型，避免内存错误。
