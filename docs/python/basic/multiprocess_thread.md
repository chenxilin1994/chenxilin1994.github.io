---
layout: doc
title: Python正则表达式完全指南
editLink: true
---


# Python多线程与多进程


## 进程和线程的概念


**进程**，是操作系统分配资源的最小单位
**线程**，是操作系统调度的最小单位

一个应用程序至少包括一个进程，一个进程最少包括一个线程。线程的尺度更小。
每个进程在执行过程中拥有独立的内存单元。一个进程中的多个线程在执行过程中共享内存。


**以下是一个生动的例子：**
- 计算机的核心是CPU，它承担了所有的计算任务。它就像一座工厂，时刻在运行。
- 假定工厂的电力有限，一次只能供给一个车间使用。也就是说，一个车间开工的时候，其他车间必须停工。意味着，一个CPU一次只能运行一个任务。多核的CPU就像有了多个发电厂，使得多工厂（多进程）得以实现。
- 进程就好比工厂的车间，它代表CPU所能处理的单个任务。任一时刻，CPU总是运行一个进程，其他进程处于非运行状态。
- 一个车间里，可以有多个工人，他们协同完成一个任务。
- 线程就好比车间里的工人，一个进程可以包括多个线程。
- 车间的空间是工人们共享的，比如许多房间是每个工人都可以进出的。这象征着一个进程的内存空间是共享的，每个线程都可以使用这些共享内存。
- 可是每个房间大小不同，有些房间只能容纳一个人，比如厕所。里面有人的时候，其他人就不能进去了。这代表着一个线程使用某些共享内存的时候，其他线程必须等它结束，才能使用这一块内存。
- 一个防止他人进入的简单方法，就是在门口加一把锁，先到的人锁上门，后到的人看到上锁，就在门口排队，等锁打开再进去。这就叫**互斥锁**（Mutual exclusion），防止多个线程同时读写某一块内存区域。
- 还有些房间可以同时容纳n人，比如厨房，也就是说，如果人数大于n，多出来的人只能在外面等着。就好比某些内存区域，只能供给固定数目的线程使用。
- 这时的解决方法，就是在门口挂n把钥匙。进去的人就取一把钥匙，出来时再把钥匙挂回原处。后到的人发现钥匙架空了，就知道必须在门口排队等着，这个做法就叫**信号量**（Semaphore），用于保证多个线程不会相互冲突。
- 不难看出，mutex是semaphore的一种特殊情况（n=1时）。也就是说，完全可以用后者替代前者。但是因为mutex比较简单且效率高，所以在必须保证资源独占的情况下，还是采用mutex。

## Python的多进程编程与multiprocess模块

python的多进程编程主要依靠multiprocess模块。我们先对比两段代码，看看多进程编程的优势。我们模拟了一个非常耗时的任务，计算8的20次方，为了使这个任务显得更耗时，我们还让它sleep 2秒。第一段代码是单进程计算(代码如下所示)，我们按顺序执行代码，重复计算2次，并打印出总共耗时。

```python
import time
import os

def long_time_task():
    print("当前进程：{}".format(os.getpid()))
    time.sleep(2)
    print("结果：{}".format(8 ** 20))

if __name__ == "__main__":
    print("当前母进程：{}".format(os.getpid()))
    start = time.time()
    for i in range(2):
        long_time_task()
    end = time.time()
    print("总耗时：{}".format(end - start))
```

输出结果：
```text
当前母进程：64760
当前进程：64760
结果：1152921504606846976
当前进程：64760
结果：1152921504606846976
总耗时：4.001052379608154
```

总耗时4秒，至始至终只有一个进程64760。

```python
from multiprocessing import Process
import os
import time

def long_time_task(i):
    print("子进程: {} - 任务{}".format(os.getpid(), i))
    time.sleep(2)
    print("结果: {}".format(8 ** 20))

if __name__ == '__main__':
    print("母进程: {}".format(os.getpid()))
    start = time.time()
    p1 = Process(target=long_time_task, args=(1,))
    p2 = Process(target=long_time_task, args=(2,))
    print("等待所有子进程完成。")
    p1.start()
    p2.start()
    p1.join()
    p2.join()
    end = time.time()
    print("总耗时: {}".format(end - start))
```

```text
母进程: 66424
等待所有子进程完成。
子进程: 75920 - 任务1
子进程: 24424 - 任务2
结果: 1152921504606846976
结果: 1152921504606846976
总耗时: 2.12857723236084
```

耗时变为2秒，时间减了一半，可见并发执行的时间明显比顺序执行要快很多。你还可以看到尽管我们只创建了两个进程，可实际运行中却包含里1个母进程和2个子进程。**之所以我们使用join()方法就是为了让母进程阻塞，等待子进程都完成后才打印出总共耗时，否则输出时间只是母进程执行的时间**。


**知识点:**
- 新创建的进程与进程的切换都是要耗资源的，所以平时工作中进程数不能开太大。
- 同时可以运行的进程数一般受制于CPU的核数。
- 除了使用Process方法，我们还可以使用Pool类创建多进程。

### 利用multiprocess模块的Pool类创建多进程

很多时候系统都需要创建多个进程以提高CPU的利用率，当数量较少时，可以手动生成一个个Process实例。当进程数量很多时，或许可以利用循环，但是这需要程序员手动管理系统中并发进程的数量，有时会很麻烦。这时进程池Pool就可以发挥其功效了。可以通过传递参数限制并发进程的数量，默认值为CPU的核数。

Pool类可以提供指定数量的进程供用户调用，当有新的请求提交到Pool中时，如果进程池还没有满，就会创建一个新的进程来执行请求。如果池满，请求就会告知先等待，直到池中有进程结束，才会创建新的进程来执行这些请求。


下面介绍一下multiprocessing 模块下的Pool类的几个方法：

1. `apply_async`

函数原型：`apply_async(func[, args=()[, kwds={}[, callback=None]]])`

其作用是向进程池提交需要执行的函数及参数， 各个进程采用非阻塞（异步）的调用方式，即每个子进程只管运行自己的，不管其它进程是否已经完成。这是默认方式。

2. `map()`

函数原型：`map(func, iterable[, chunksize=None])`

Pool类中的map方法，与内置的map函数用法行为基本一致，它会使进程阻塞直到结果返回。 注意：虽然第二个参数是一个迭代器，但在实际使用中，必须在整个队列都就绪后，程序才会运行子进程。

3. `map_async()`

函数原型：`map_async(func, iterable[, chunksize[, callback]])`

与map用法一致，但是它是非阻塞的。其有关事项见apply_async。

4. `close()`

关闭进程池（pool），使其不在接受新的任务。

5. `terminate()`

结束工作进程，不在处理未处理的任务。

6. `join()`

主进程阻塞等待子进程的退出， join方法要在close或terminate之后使用。

下例是一个简单的multiprocessing.Pool类的实例。因为我的CPU是16核的，一次最多可以同时运行16个进程，所以我开启了一个容量为16的进程池。16个进程需要计算17次，你可以想象16个进程并行16次计算任务后，还剩一次计算任务没有完成，系统会等待16个进程完成后重新安排一个进程来计算。
```python
from multiprocessing import Pool, cpu_count
import os
import time


def long_time_task(i) -> None:
    print("子进程: {} - 任务{}".format(os.getpid(), i))
    time.sleep(2)
    print("结果: {}".format(8 ** 20))


if __name__ == '__main__':
    print("CPU内核数:{}".format(cpu_count()))
    print("当前母进程: {}".format(os.getpid()))
    start = time.time()
    p = Pool(cpu_count())
    for i in range(cpu_count() + 1):
        p.apply_async(long_time_task, args=(i,))
    print("等待所有子进程完成。")
    p.close()
    p.join()
    end = time.time()
    print("总耗时: {}".format(end - start))
```

```text
CPU内核数:16
当前母进程: 83696
等待所有子进程完成。
子进程: 45460 - 任务0
子进程: 80996 - 任务1
子进程: 57980 - 任务2
子进程: 81736 - 任务3
子进程: 37456 - 任务4
子进程: 77636 - 任务5
子进程: 37708 - 任务6
子进程: 75140 - 任务7
子进程: 66860 - 任务8
子进程: 24952 - 任务9
子进程: 7516 - 任务10
子进程: 81320 - 任务11
子进程: 31144 - 任务12
子进程: 74828 - 任务13
子进程: 75924 - 任务14
子进程: 31448 - 任务15
结果: 1152921504606846976
子进程: 45460 - 任务16
结果: 1152921504606846976
结果: 1152921504606846976
结果: 1152921504606846976
结果: 1152921504606846976
结果: 1152921504606846976
结果: 1152921504606846976
结果: 1152921504606846976
结果: 1152921504606846976
结果: 1152921504606846976
结果: 1152921504606846976
结果: 1152921504606846976
结果: 1152921504606846976
结果: 1152921504606846976
结果: 1152921504606846976
结果: 1152921504606846976
结果: 1152921504606846976
结果: 1152921504606846976
结果: 1152921504606846976
结果: 1152921504606846976
结果: 1152921504606846976
结果: 1152921504606846976
结果: 1152921504606846976
结果: 1152921504606846976
结果: 1152921504606846976
结果: 1152921504606846976
结果: 1152921504606846976
结果: 1152921504606846976
结果: 1152921504606846976
结果: 1152921504606846976
结果: 1152921504606846976
结果: 1152921504606846976
总耗时: 4.239032506942749
```

**知识点：**
- 对Pool对象调用join方法会等待所有子进程执行完毕，调用join之前必须先调用close或terminate方法，让其不再接受新的Process

输出结果如下所示，17个任务（每个任务大约耗时2秒）使用多进程并行计算只需4秒, 可见并行计算优势还是很明显的。

相信大家都知道python解释器中存在GIL([全局解释器锁](TODO:添加链接)), 它的作用就是保证同一时刻只有一个线程可以执行代码。由于GIL的存在，很多人认为python中的多线程其实并不是真正的多线程，如果想要充分地使用多核CPU的资源，在python中大部分情况需要使用多进程。然而这并意味着python多线程编程没有意义。

### 多进程间的数据共享与通信

通常，进程之间是相互独立的，每个进程都有独立的内存。通过共享内存(nmap模块)，进程之间可以共享对象，使多个进程可以访问同一个变量(地址相同，变量名可能不同)。多进程共享资源必然会导致进程间相互竞争，所以应该尽最大可能防止使用共享状态。还有一种方式就是使用队列queue来实现不同进程间的通信或数据共享，这一点和多线程编程类似。


```python
# 下例这段代码中中创建了2个独立进程，一个负责写(pw), 一个负责读(pr), 实现了共享一个队列queue。

from multiprocessing import Process, Queue
import os, time, random

def write(q: Queue):
    print("Process to write: {}".format(os.getpid()))
    for value in "ABCDE":
        print("Put %s to queue..." % value)
        q.put(value)
        time.sleep(random.random())


def read(q: Queue):
    print("Process to read: {}".format(os.getpid()))
    while True:
        value = q.get(True)
        print("Get %s from queue." % value)


if __name__ == '__main__':
    # 父进程创建Queue，并传递给各个子进程
    q = Queue()
    pw = Process(target=write, args=(q,))
    pr = Process(target=read, args=(q,))
    # 启动子进程pw，写入:
    pw.start()
    # 启动子进程pr，读取:
    pr.start()
    # 等待pw结束:
    pw.join()
    # pr进程里是死循环，无法等待其结束，只能强行终止:
    pr.terminate()
```
```text
Process to write: 20628
Put A to queue...
Process to read: 51764
Get A from queue.
Put B to queue...
Get B from queue.
Put C to queue...
Get C from queue.
Put D to queue...
Get D from queue.
Put E to queue...
Get E from queue.
```

## Python的多线程编程与threading模块

python 3中的多进程编程主要依靠threading模块。创建新线程与创建新进程的方法非常类似。threading.Thread方法可以接收两个参数, 第一个是target，一般指向函数名，第二个时args，需要向函数传递的参数。对于创建的新线程，调用start()方法即可让其开始。我们还可以使用current_thread().name打印出当前线程的名字。 下例中我们使用多线程技术重构之前的计算代码。

```python
import threading
import time

def long_time_task(i):
    print("当前子进程：{} - 任务{}".format(threading.current_thread().name, i))
    time.sleep(2)
    print("结果：{}".format(8**20))


if __name__ == '__main__':
    start = time.time()
    print("这是主线程：{}".format(threading.current_thread().name))
    t1 = threading.Thread(target=long_time_task, args=(1,))
    t2 = threading.Thread(target=long_time_task, args=(2,))
    t1.start()
    t2.start()
    
    end = time.time()
    print("总耗时：{}秒".format((end-start)))
```
```text
这是主线程：MainThread
当前子进程：Thread-1 (long_time_task) - 任务1
当前子进程：Thread-2 (long_time_task) - 任务2
总耗时：0.002134084701538086秒
结果：1152921504606846976
结果：1152921504606846976
```

为什么总耗时居然是0秒? 我们可以明显看到主线程和子线程其实是独立运行的，主线程根本没有等子线程完成，而是自己结束后就打印了消耗时间。主线程结束后，子线程仍在独立运行，这显然不是我们想要的。

如果要实现主线程和子线程的同步，我们必需使用join方法（代码如下所示)。

```python
import threading
import time

def long_time_task(i):
    print("当前子进程：{} - 任务{}".format(threading.current_thread().name, i))
    time.sleep(2)
    print("结果：{}".format(8**20))


if __name__ == '__main__':
    start = time.time()
    print("这是主线程：{}".format(threading.current_thread().name))
    t1 = threading.Thread(target=long_time_task, args=(1,))
    t2 = threading.Thread(target=long_time_task, args=(2,))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    
    end = time.time()
    print("总耗时：{}秒".format((end-start)))
```
```text
这是主线程：MainThread
当前子进程：Thread-1 (long_time_task) - 任务1
当前子进程：Thread-2 (long_time_task) - 任务2
结果：1152921504606846976结果：1152921504606846976

总耗时：2.002108573913574秒
```

修改代码后的输出如下所示。这时你可以看到主线程在等子线程完成后才答应出总消耗时间(2秒)，比正常顺序执行代码(4秒)还是节省了不少时间。

当我们设置多线程时，主线程会创建多个子线程，在python中，默认情况下主线程和子线程独立运行互不干涉。**如果希望让主线程等待子线程实现线程的同步，我们需要使用join()方法**。如果我们希望一个主线程结束时不再执行子线程，我们应该怎么办呢? 我们可以使用t.setDaemon(True)，代码如下所示。

### 通过继承Thread类重写run方法创建新线程

除了使用Thread()方法创建新的线程外，我们还可以通过继承Thread类重写run方法创建新的线程，这种方法更灵活。下例中我们自定义的类为MyThread, 随后我们通过该类的实例化创建了2个子线程。

```python
"""
场景描述
主线程：扮演蜂巢，控制整个采蜜任务的开始和结束。

子线程：扮演蜜蜂，每只蜜蜂独立工作，每秒采集一次蜂蜜，直到蜂巢发出停止信号。
"""

import threading
import time

from threading import Event


class Bee(threading.Thread):

    def __init__(self, name, hive_event: Event):
        super().__init__()  # 必须调用父类初始化
        self.name = name
        self.hive_event: Event = hive_event

    def run(self):
        """重写run方法, 定义蜜蜂的行为"""
        print(f"{self.name}出发采蜜啦！")
        honey_count = 0
        while not self.hive_event.is_set():
            print(f"{self.name}采集到1滴蜂蜜~")
            honey_count += 1
            time.sleep(1)  # 模拟采蜜耗时，1秒1滴
        # 当事件触发时，退出循环
        print(f"{self.name}回家啦！共采集了{honey_count}滴蜂蜜。")

if __name__ == '__main__':
    # 创建一个事件对象，用于控制蜜蜂停止
    stop_event = threading.Event()
    # 创建3只蜜蜂线程
    bees = [
        Bee("小黄蜂", stop_event),
        Bee("闪电蜂", stop_event),
        Bee("小先蜂", stop_event)
    ]
    # 启动所有蜜蜂线程
    for bee in bees:
        bee.start()

    # 主线程（蜂巢）等待5秒后发出信号，停止采蜜，回家
    time.sleep(5)
    stop_event.set()  # 设置事件，通知蜜蜂停止

    # 等待所有蜜蜂线程结束
    for bee in bees:
        bee.join()

    print("蜂巢关闭！")

```

```text
小黄蜂出发采蜜啦！
闪电蜂出发采蜜啦！小黄蜂采集到1滴蜂蜜~小先蜂出发采蜜啦！


闪电蜂采集到1滴蜂蜜~小先蜂采集到1滴蜂蜜~

小黄蜂采集到1滴蜂蜜~
小先蜂采集到1滴蜂蜜~闪电蜂采集到1滴蜂蜜~

小黄蜂采集到1滴蜂蜜~闪电蜂采集到1滴蜂蜜~

小先蜂采集到1滴蜂蜜~
小先蜂采集到1滴蜂蜜~小黄蜂采集到1滴蜂蜜~闪电蜂采集到1滴蜂蜜~


闪电蜂采集到1滴蜂蜜~小黄蜂采集到1滴蜂蜜~小先蜂采集到1滴蜂蜜~


闪电蜂回家啦！共采集了5滴蜂蜜。
小黄蜂回家啦！共采集了5滴蜂蜜。小先蜂回家啦！共采集了5滴蜂蜜。

蜂巢关闭！
```

### 不同线程间的数据共享


一个进程所含的不同线程间共享内存，这就意味着任何一个变量都可以被任何一个线程修改，因此线程之间共享数据最大的危险在于多个线程同时改一个变量，把内容给改乱了。如果不同线程间有共享的变量，其中一个方法就是在修改前给其上一把锁lock，确保一次只有一个线程能修改它。threading.lock()方法可以轻易实现对一个共享变量的锁定，修改完后release供其它线程使用。比如下例中账户余额balance是一个共享变量，使用lock可以使其不被改乱。

```python
"""
场景升级：蜜蜂协作采蜜
共享变量：所有蜜蜂共同为蜂巢采集蜂蜜，累计到 total_honey。

锁的作用：确保每次只有一个线程能修改 total_honey，避免数据混乱。
"""
import threading
import time

# 继承 Thread 类，自定义蜜蜂线程
class Bee(threading.Thread):
    def __init__(self, name, hive_event, total_honey, lock):
        super().__init__()
        self.name = name
        self.hive_event = hive_event  # 停止信号事件
        self.total_honey = total_honey  # 共享变量：总蜂蜜量
        self.lock = lock              # 共享锁

    def run(self):
        print(f"{self.name} 出发采蜜啦！")
        while not self.hive_event.is_set():
            # 模拟采蜜耗时（不加锁的操作可以并行）
            time.sleep(0.5)  # 假设找到花朵需要 0.5 秒
            
            # 修改共享变量前加锁
            with self.lock:
                self.total_honey[0] += 1  # 通过列表存储以传递引用
                print(f"{self.name} 将 1 滴蜂蜜存入蜂巢，当前总量：{self.total_honey[0]}")

        print(f"{self.name} 回家啦！")

if __name__ == "__main__":
    stop_event = threading.Event()
    total_honey = [0]  # 使用列表传递引用，确保共享
    lock = threading.Lock()  # 创建锁对象

    # 创建 3 只蜜蜂，并传入共享变量和锁
    bees = [
        Bee("小黄蜂", stop_event, total_honey, lock),
        Bee("胖蜜蜂", stop_event, total_honey, lock),
        Bee("闪电蜂", stop_event, total_honey, lock),
    ]

    for bee in bees:
        bee.start()

    # 主线程等待 3 秒后发出停止信号
    time.sleep(3)
    print("\n蜂巢：天黑啦，所有蜜蜂回家！")
    stop_event.set()

    for bee in bees:
        bee.join()

    print(f"最终蜂蜜总量：{total_honey[0]} 滴")
```

```text
小黄蜂 出发采蜜啦！
胖蜜蜂 出发采蜜啦！闪电蜂 出发采蜜啦！

小黄蜂 将 1 滴蜂蜜存入蜂巢，当前总量：1
胖蜜蜂 将 1 滴蜂蜜存入蜂巢，当前总量：2
闪电蜂 将 1 滴蜂蜜存入蜂巢，当前总量：3
闪电蜂 将 1 滴蜂蜜存入蜂巢，当前总量：4
胖蜜蜂 将 1 滴蜂蜜存入蜂巢，当前总量：5
小黄蜂 将 1 滴蜂蜜存入蜂巢，当前总量：6
闪电蜂 将 1 滴蜂蜜存入蜂巢，当前总量：7
小黄蜂 将 1 滴蜂蜜存入蜂巢，当前总量：8
胖蜜蜂 将 1 滴蜂蜜存入蜂巢，当前总量：9
闪电蜂 将 1 滴蜂蜜存入蜂巢，当前总量：10
小黄蜂 将 1 滴蜂蜜存入蜂巢，当前总量：11
胖蜜蜂 将 1 滴蜂蜜存入蜂巢，当前总量：12
闪电蜂 将 1 滴蜂蜜存入蜂巢，当前总量：13
胖蜜蜂 将 1 滴蜂蜜存入蜂巢，当前总量：14
小黄蜂 将 1 滴蜂蜜存入蜂巢，当前总量：15

蜂巢：天黑啦，所有蜜蜂回家！
小黄蜂 将 1 滴蜂蜜存入蜂巢，当前总量：16
小黄蜂 回家啦！胖蜜蜂 将 1 滴蜂蜜存入蜂巢，当前总量：17

胖蜜蜂 回家啦！闪电蜂 将 1 滴蜂蜜存入蜂巢，当前总量：18

闪电蜂 回家啦！
最终蜂蜜总量：18 滴
```

另一种实现不同线程间数据共享的方法就是使用消息队列queue。不像列表，queue是线程安全的，可以放心使用，见下文。

### 使用queue队列通信-经典的生产者和消费者模型

```python
import threading
import time
import queue

# 蜜蜂线程：采蜜并放入队列
class Bee(threading.Thread):
    def __init__(self, name, stop_event, honey_queue) -> None:
        super().__init__()
        self.name = name
        self.stop_event = stop_event  # 停止信号
        self.honey_queue = honey_queue  # 共享队列

    def run(self) -> None:
        print(f"{self.name} 出发采蜜啦！")
        while not self.stop_event.is_set():
            time.sleep(0.3)  # 模拟采蜜耗时
            # 将蜂蜜放入队列（线程安全）
            self.honey_queue.put(1)
            print(f"{self.name} 将 1 滴蜂蜜放入队列")
        print(f"{self.name} 回家啦！")

# 消费者线程：从队列中取蜂蜜并累加
def honey_consumer(honey_queue, total_honey, stop_event) -> None:
    while True:
        try:
            # 从队列取蜂蜜（最多等待1秒，避免永久阻塞）
            honey = honey_queue.get(timeout=1)
            if total_honey[0] >= 50:
                # 如果已达50滴，标记任务完成但不累加
                honey_queue.task_done()
                continue
            
            # 累加蜂蜜（单线程操作，无需加锁）
            total_honey[0] += honey
            print(f"蜂巢收到 1 滴蜂蜜，当前总量：{total_honey[0]}")
            
            # 检查是否达到50滴
            if total_honey[0] >= 50:
                stop_event.set()  # 触发停止信号
                print("\n蜂巢：蜂蜜已满50滴，停止采集！")
                
            honey_queue.task_done()
        except queue.Empty:
            # 如果队列为空且已触发停止，退出循环
            if stop_event.is_set():
                break

if __name__ == "__main__":
    stop_event = threading.Event()
    honey_queue = queue.Queue()  # 线程安全队列
    total_honey = [0]  # 使用列表传递引用

    # 创建3只蜜蜂
    bees = [
        Bee("小黄蜂", stop_event, honey_queue),
        Bee("胖蜜蜂", stop_event, honey_queue),
        Bee("闪电蜂", stop_event, honey_queue),
    ]

    # 启动所有蜜蜂
    for bee in bees:
        bee.start()

    # 启动消费者线程
    consumer = threading.Thread(
        target=honey_consumer,
        args=(honey_queue, total_honey, stop_event)
    )
    consumer.start()

    # 主线程等待停止信号
    while not stop_event.is_set():
        time.sleep(0.1)

    # 等待所有蜜蜂结束
    for bee in bees:
        bee.join()

    # 等待队列中的任务全部处理完成
    honey_queue.join()
    consumer.join()

    print(f"最终蜂蜜总量：{total_honey[0]} 滴")
```

```text
小黄蜂 出发采蜜啦！
胖蜜蜂 出发采蜜啦！闪电蜂 出发采蜜啦！

小黄蜂 将 1 滴蜂蜜放入队列蜂巢收到 1 滴蜂蜜，当前总量：1

闪电蜂 将 1 滴蜂蜜放入队列胖蜜蜂 将 1 滴蜂蜜放入队列蜂巢收到 1 滴蜂蜜，当前总量：2


蜂巢收到 1 滴蜂蜜，当前总量：3
蜂巢收到 1 滴蜂蜜，当前总量：4小黄蜂 将 1 滴蜂蜜放入队列

蜂巢收到 1 滴蜂蜜，当前总量：5闪电蜂 将 1 滴蜂蜜放入队列胖蜜蜂 将 1 滴蜂蜜放入队列


蜂巢收到 1 滴蜂蜜，当前总量：6
小黄蜂 将 1 滴蜂蜜放入队列胖蜜蜂 将 1 滴蜂蜜放入队列蜂巢收到 1 滴蜂蜜，当前总量：7闪电蜂 将 1 滴蜂蜜放入队列



蜂巢收到 1 滴蜂蜜，当前总量：8
蜂巢收到 1 滴蜂蜜，当前总量：9
闪电蜂 将 1 滴蜂蜜放入队列小黄蜂 将 1 滴蜂蜜放入队列蜂巢收到 1 滴蜂蜜，当前总量：10胖蜜蜂 将 1 滴蜂蜜放入队列



蜂巢收到 1 滴蜂蜜，当前总量：11
蜂巢收到 1 滴蜂蜜，当前总量：12
蜂巢收到 1 滴蜂蜜，当前总量：13小黄蜂 将 1 滴蜂蜜放入队列闪电蜂 将 1 滴蜂蜜放入队列


蜂巢收到 1 滴蜂蜜，当前总量：14
蜂巢收到 1 滴蜂蜜，当前总量：15胖蜜蜂 将 1 滴蜂蜜放入队列

闪电蜂 将 1 滴蜂蜜放入队列小黄蜂 将 1 滴蜂蜜放入队列

蜂巢收到 1 滴蜂蜜，当前总量：16
蜂巢收到 1 滴蜂蜜，当前总量：17
蜂巢收到 1 滴蜂蜜，当前总量：18胖蜜蜂 将 1 滴蜂蜜放入队列

蜂巢收到 1 滴蜂蜜，当前总量：19小黄蜂 将 1 滴蜂蜜放入队列

蜂巢收到 1 滴蜂蜜，当前总量：21
小黄蜂 将 1 滴蜂蜜放入队列闪电蜂 将 1 滴蜂蜜放入队列蜂巢收到 1 滴蜂蜜，当前总量：22


胖蜜蜂 将 1 滴蜂蜜放入队列
蜂巢收到 1 滴蜂蜜，当前总量：23
蜂巢收到 1 滴蜂蜜，当前总量：24
小黄蜂 将 1 滴蜂蜜放入队列闪电蜂 将 1 滴蜂蜜放入队列蜂巢收到 1 滴蜂蜜，当前总量：25


蜂巢收到 1 滴蜂蜜，当前总量：26
蜂巢收到 1 滴蜂蜜，当前总量：27
胖蜜蜂 将 1 滴蜂蜜放入队列
闪电蜂 将 1 滴蜂蜜放入队列小黄蜂 将 1 滴蜂蜜放入队列蜂巢收到 1 滴蜂蜜，当前总量：28


蜂巢收到 1 滴蜂蜜，当前总量：29
蜂巢收到 1 滴蜂蜜，当前总量：30胖蜜蜂 将 1 滴蜂蜜放入队列

小黄蜂 将 1 滴蜂蜜放入队列蜂巢收到 1 滴蜂蜜，当前总量：31闪电蜂 将 1 滴蜂蜜放入队列胖蜜蜂 将 1 滴蜂蜜放入队列



蜂巢收到 1 滴蜂蜜，当前总量：32
蜂巢收到 1 滴蜂蜜，当前总量：33
小黄蜂 将 1 滴蜂蜜放入队列闪电蜂 将 1 滴蜂蜜放入队列蜂巢收到 1 滴蜂蜜，当前总量：34胖蜜蜂 将 1 滴蜂蜜放入队列



蜂巢收到 1 滴蜂蜜，当前总量：35
蜂巢收到 1 滴蜂蜜，当前总量：36
蜂巢收到 1 滴蜂蜜，当前总量：37小黄蜂 将 1 滴蜂蜜放入队列胖蜜蜂 将 1 滴蜂蜜放入队列闪电蜂 将 1 滴蜂蜜放入队列



蜂巢收到 1 滴蜂蜜，当前总量：38
蜂巢收到 1 滴蜂蜜，当前总量：39
胖蜜蜂 将 1 滴蜂蜜放入队列小黄蜂 将 1 滴蜂蜜放入队列蜂巢收到 1 滴蜂蜜，当前总量：40
闪电蜂 将 1 滴蜂蜜放入队列


蜂巢收到 1 滴蜂蜜，当前总量：41
蜂巢收到 1 滴蜂蜜，当前总量：42
小黄蜂 将 1 滴蜂蜜放入队列蜂巢收到 1 滴蜂蜜，当前总量：43胖蜜蜂 将 1 滴蜂蜜放入队列


蜂巢收到 1 滴蜂蜜，当前总量：44
闪电蜂 将 1 滴蜂蜜放入队列蜂巢收到 1 滴蜂蜜，当前总量：45

小黄蜂 将 1 滴蜂蜜放入队列蜂巢收到 1 滴蜂蜜，当前总量：46胖蜜蜂 将 1 滴蜂蜜放入队列


蜂巢收到 1 滴蜂蜜，当前总量：47
闪电蜂 将 1 滴蜂蜜放入队列蜂巢收到 1 滴蜂蜜，当前总量：48

蜂巢收到 1 滴蜂蜜，当前总量：49胖蜜蜂 将 1 滴蜂蜜放入队列小黄蜂 将 1 滴蜂蜜放入队列


蜂巢收到 1 滴蜂蜜，当前总量：50

蜂巢：蜂蜜已满50滴，停止采集！
闪电蜂 将 1 滴蜂蜜放入队列
闪电蜂 回家啦！
小黄蜂 将 1 滴蜂蜜放入队列胖蜜蜂 将 1 滴蜂蜜放入队列

胖蜜蜂 回家啦！小黄蜂 回家啦！

最终蜂蜜总量：50 滴
```

## Python多进程和多线程哪个快?

由于GIL的存在，很多人认为Python多进程编程更快，针对多核CPU，理论上来说也是采用多进程更能有效利用资源。

对CPU密集型代码(比如循环计算) - 多进程效率更高
对IO密集型代码(比如文件操作，网络爬虫) - 多线程效率更高。

为什么是这样呢？其实也不难理解。对于IO密集型操作，大部分消耗时间其实是等待时间，在等待时间中CPU是不需要工作的，那你在此期间提供双CPU资源也是利用不上的，相反对于CPU密集型代码，2个CPU干活肯定比一个CPU快很多。
那么为什么多线程会对IO密集型代码有用呢？这时因为python碰到等待会释放GIL供新的线程使用，实现了线程间的切换。