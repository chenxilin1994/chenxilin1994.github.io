

# 并发与异步编程

## 41. 如何在Python中使用多进程池（Pool）？
详细答案  
- 作用：  
  `multiprocessing.Pool` 管理进程池，自动分配任务到多个进程，充分利用多核CPU。  
- 核心方法：  
  - `map(func, iterable)`：将可迭代对象分块映射到函数，阻塞直到所有任务完成。  
  - `apply_async(func, args)`：异步提交单个任务，返回 `AsyncResult` 对象。  
  - `close()`：停止接受新任务。  
  - `join()`：等待所有进程退出。  

代码示例  
```python
from multiprocessing import Pool
import time

def square(x):
    time.sleep(0.5)
    return x  2

if __name__ == '__main__':
    with Pool(4) as pool:  # 创建4个进程的池
        results = pool.map(square, range(10))  # 并行计算平方
    print(results)  # [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

    # 异步提交任务
    pool = Pool(4)
    async_result = pool.apply_async(square, (10,))
    print(async_result.get())  # 100（阻塞等待结果）
    pool.close()
    pool.join()
```



## 42. 解释Python中的异步生成器（Async Generator）及其用途。
详细答案  
- 异步生成器：  
  - 使用 `async def` 定义，包含 `yield` 语句。  
  - 生成值时可暂停并允许其他异步任务执行。  
- 用途：  
  - 流式处理大数据（如分批读取数据库）。  
  - 在异步循环中逐个生成结果。  

代码示例  
```python
import asyncio

async def async_gen():
    for i in range(3):
        await asyncio.sleep(0.5)
        yield i

async def main():
    async for num in async_gen():
        print(num)  # 输出0, 1, 2（间隔0.5秒）

asyncio.run(main())
```



## 43. 如何用 `asyncio` 实现并发HTTP请求？
详细答案  
- 步骤：  
  1. 使用 `aiohttp` 库发送异步HTTP请求。  
  2. 创建多个协程任务并通过 `asyncio.gather()` 并行执行。  

代码示例  
```python
import aiohttp
import asyncio

async def fetch(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.text()

async def main():
    urls = ["http://example.com"] * 5
    tasks = [fetch(url) for url in urls]
    results = await asyncio.gather(*tasks)
    print(f"获取{len(results)}个页面")

asyncio.run(main())
```



## 44. 解释线程局部存储（Thread-Local Data）及其应用场景。
详细答案  
- 线程局部存储：  
  - 每个线程独立访问的数据副本，避免多线程竞争。  
  - 使用 `threading.local()` 创建线程局部对象。  

- 应用场景：  
  - Web请求处理中保存用户会话（如Flask的请求上下文）。  

代码示例  
```python
import threading
from threading import Thread

local_data = threading.local()

def show_data():
    print(threading.current_thread().name, local_data.value)

def worker(value):
    local_data.value = value  # 每个线程独立赋值
    show_data()

threads = [
    Thread(target=worker, args=("A",)),
    Thread(target=worker, args=("B",))
]
for t in threads:
    t.start()
for t in threads:
    t.join()
# 输出：
# Thread-1 A
# Thread-2 B
```



## 45. 如何在Python中实现生产者-消费者模式？
详细答案  
- 模式结构：  
  - 生产者：生成数据并放入队列。  
  - 消费者：从队列取出数据并处理。  
  - 队列：线程安全的缓冲区（`queue.Queue`）。  

代码示例  
```python
import threading
import queue
import time

def producer(q):
    for i in range(5):
        q.put(i)
        print(f"生产 {i}")
        time.sleep(0.1)

def consumer(q):
    while True:
        item = q.get()
        if item is None:  # 终止信号
            break
        print(f"消费 {item}")
        q.task_done()

q = queue.Queue()
threads = [
    threading.Thread(target=producer, args=(q,)),
    threading.Thread(target=consumer, args=(q,))
]
for t in threads:
    t.start()
for t in threads:
    t.join(timeout=1)
q.put(None)  # 发送终止信号
```



## 46. 解释Python中的 `Future` 和 `Promise` 概念。
详细答案  
- Future：  
  - 表示异步操作的未完成结果，提供结果状态查询和回调注册。  
  - `concurrent.futures.Future`（线程/进程）和 `asyncio.Future`（协程）。  

- Promise：  
  - 类似 `Future`，但结果由外部显式设置（如JavaScript中的Promise）。  
  - Python中通常直接用 `Future`。  

代码示例  
```python
from concurrent.futures import ThreadPoolExecutor

def task(n):
    return n * 2

with ThreadPoolExecutor() as executor:
    future = executor.submit(task, 5)
    print(future.result())  # 10（阻塞直到结果就绪）
```



## 47. 如何处理异步编程中的异常？
详细答案  
- 方法：  
  1. 在协程内使用 `try/except` 捕获异常。  
  2. 通过 `asyncio.gather(return_exceptions=True)` 避免异常传播。  

代码示例  
```python
import asyncio

async def risky_task():
    raise ValueError("出错")

async def main():
    try:
        await risky_task()
    except ValueError as e:
        print(f"捕获异常：{e}")

    # 批量处理任务异常
    tasks = [risky_task() for _ in range(3)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for r in results:
        if isinstance(r, Exception):
            print("任务失败：", r)

asyncio.run(main())
```



## 48. 解释协程（Coroutine）与生成器（Generator）的区别。
详细答案  
- 生成器：  
  - 用于生成迭代序列，通过 `yield` 产生值。  
  - 单向通信（调用者获取值）。  

- 协程：  
  - 通过 `yield` 接收和发送数据（双向通信）。  
  - 用于并发编程，支持 `async/await` 语法（Python 3.5+）。  

代码示例  
```python
# 生成器
def gen():
    yield 1
    yield 2

# 协程（旧式，基于生成器）
def old_coroutine():
    x = yield
    yield x + 1

c = old_coroutine()
next(c)
print(c.send(5))  # 6

# 新式协程
async def new_coroutine():
    await asyncio.sleep(1)
    return "完成"
```



## 49. 如何在多进程中使用队列（Queue）进行通信？
详细答案  
- 方法：  
  - 使用 `multiprocessing.Queue`（跨进程安全）。  
  - 通过 `put()` 和 `get()` 传递数据。  

代码示例  
```python
from multiprocessing import Process, Queue

def worker(q):
    q.put("子进程消息")

if __name__ == '__main__':
    q = Queue()
    p = Process(target=worker, args=(q,))
    p.start()
    print(q.get())  # 输出"子进程消息"
    p.join()
```



## 50. 解释Python中的上下文变量（Context Variables）及其用途。
详细答案  
- 上下文变量：  
  - 用于在异步任务或线程间传递上下文数据（如请求ID）。  
  - 通过 `contextvars` 模块实现，每个协程有独立副本。  

代码示例  
```python
import contextvars
import asyncio

request_id = contextvars.ContextVar('request_id')

async def handle_request(id):
    request_id.set(id)
    print(f"处理请求 {request_id.get()}")

async def main():
    await asyncio.gather(
        handle_request(1),
        handle_request(2)
    )  # 输出：处理请求1 → 处理请求2（各自独立）

asyncio.run(main())
```