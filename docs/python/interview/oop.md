
# 面向对象编程（OOP）

## 31. 类变量（Class Variable）和实例变量（Instance Variable）的区别是什么？
详细答案  
- 类变量：  
  - 定义在类内部且在方法外，所有实例共享。  
  - 用于存储类级别的数据（如配置、全局计数器）。  
  - 通过类名或实例访问（若实例未覆盖同名变量）。  

- 实例变量：  
  - 定义在 `__init__` 或实例方法中，每个实例独立。  
  - 用于存储对象特有的状态。  

代码示例  
```python
class Employee:
    company = "Tech Corp"  # 类变量

    def __init__(self, name):
        self.name = name    # 实例变量

e1 = Employee("Alice")
e2 = Employee("Bob")

print(e1.company)  # "Tech Corp"（类变量）
Employee.company = "New Corp"
print(e2.company)  # "New Corp"（所有实例共享）

e1.company = "Local Corp"  # 为e1创建同名实例变量
print(e1.company)  # "Local Corp"（覆盖类变量）
print(Employee.company)  # "New Corp"（类变量未变）
```



## 32. 什么是方法解析顺序（MRO）？Python如何确定它？
详细答案  
- MRO：定义类继承关系中方法查找的顺序（解决多重继承的方法冲突）。  
- C3线性化算法：Python使用此算法计算MRO，规则为：  
  1. 子类优先于父类。  
  2. 多个父类按定义顺序保留。  
  3. 所有父类的MRO合并时保持单调性。  

代码示例  
```python
class A:
    def show(self):
        print("A")

class B(A):
    def show(self):
        print("B")

class C(A):
    def show(self):
        print("C")

class D(B, C):
    pass

print(D.mro())  # [D, B, C, A, object]
d = D()
d.show()        # 输出 "B"（按MRO顺序查找）
```



## 33. `super()` 函数的作用是什么？在多重继承中如何工作？
详细答案  
- 作用：  
  - 调用父类的方法，避免硬编码父类名称。  
  - 在继承链中按MRO顺序传递方法调用。  

- 多重继承中的行为：  
  - `super()` 根据当前类的MRO动态决定下一个调用的类。  

代码示例  
```python
class A:
    def __init__(self):
        print("A")

class B(A):
    def __init__(self):
        super().__init__()
        print("B")

class C(A):
    def __init__(self):
        super().__init__()
        print("C")

class D(B, C):
    def __init__(self):
        super().__init__()
        print("D")

d = D()  # 输出顺序：A → C → B → D（遵循MRO: D → B → C → A → object）
```



## 34. 如何实现单例模式（Singleton Pattern）？至少提供两种方法。
详细答案  
- 方法1：使用元类  
  - 控制类的实例化过程，确保仅创建一个实例。  

- 方法2：使用装饰器  
  - 包装类，通过闭包管理实例。  

代码示例  
```python
# 方法1：元类
class SingletonMeta(type):
    _instances = {}
    def __call__(cls, *args, kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, kwargs)
        return cls._instances[cls]

class Database(metaclass=SingletonMeta):
    pass

d1 = Database()
d2 = Database()
print(d1 is d2)  # True

# 方法2：装饰器
def singleton(cls):
    instances = {}
    def wrapper(*args, kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, kwargs)
        return instances[cls]
    return wrapper

@singleton
class Logger:
    pass

l1 = Logger()
l2 = Logger()
print(l1 is l2)  # True
```



## 35. `__new__` 和 `__init__` 方法的区别是什么？
详细答案  
- `__new__`：  
  - 静态方法，负责创建并返回类的实例（通常调用 `object.__new__`）。  
  - 必须返回实例，否则 `__init__` 不会执行。  

- `__init__`：  
  - 实例方法，负责初始化实例的属性。  
  - 无返回值。  

代码示例  
```python
class MyClass:
    def __new__(cls, *args, kwargs):
        print("创建实例")
        instance = super().__new__(cls)
        return instance  # 必须返回实例

    def __init__(self, value):
        print("初始化实例")
        self.value = value

obj = MyClass(10)  # 输出顺序：创建实例 → 初始化实例
```



## 36. 什么是抽象基类（Abstract Base Class, ABC）？如何定义？
详细答案  
- 抽象基类：  
  - 定义接口规范，强制子类实现特定方法。  
  - 不能直接实例化，通过 `abc` 模块实现。  

- 定义方法：  
  1. 继承 `abc.ABC`。  
  2. 使用 `@abc.abstractmethod` 装饰抽象方法。  

代码示例  
```python
from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def area(self):
        pass

class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius

    def area(self):
        return 3.14 * self.radius  2

# shape = Shape()  # 报错：无法实例化抽象类
circle = Circle(5)
print(circle.area())  # 78.5
```



## 37. 什么是混入类（Mixin Class）？它的设计原则是什么？
详细答案  
- Mixin类：  
  - 提供特定功能的类，用于多重继承，增强代码复用。  
  - 不单独使用，而是作为父类之一与其他类组合。  

- 设计原则：  
  1. 功能单一（如日志、序列化）。  
  2. 不定义 `__init__` 方法（避免与主类冲突）。  

代码示例  
```python
class JSONSerializableMixin:
    def to_json(self):
        import json
        return json.dumps(self.__dict__)

class User(JSONSerializableMixin):
    def __init__(self, name):
        self.name = name

user = User("Alice")
print(user.to_json())  # {"name": "Alice"}
```



## 38. 如何实现对象的上下文管理（`with`语句）？
详细答案  
- 协议方法：  
  1. `__enter__()`：返回上下文资源（如文件对象）。  
  2. `__exit__(exc_type, exc_val, traceback)`：处理清理操作（如关闭文件）。  

代码示例  
```python
class FileHandler:
    def __init__(self, filename, mode):
        self.filename = filename
        self.mode = mode

    def __enter__(self):
        self.file = open(self.filename, self.mode)
        return self.file

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()
        if exc_type:
            print(f"异常发生：{exc_val}")

with FileHandler("test.txt", "w") as f:
    f.write("Hello")
```



## 39. 解释描述符（Descriptor）协议及其应用场景。
详细答案  
- 描述符协议：  
  - 实现 `__get__`、`__set__` 或 `__delete__` 方法的对象。  
  - 用于控制属性访问（如类型检查、惰性加载）。  

- 应用场景：  
  1. 属性验证（如确保数值范围）。  
  2. 实现 `@property` 装饰器的底层机制。  

代码示例  
```python
class PositiveNumber:
    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, instance, owner):
        return instance.__dict__[self.name]

    def __set__(self, instance, value):
        if value <= 0:
            raise ValueError("必须为正数")
        instance.__dict__[self.name] = value

class Order:
    quantity = PositiveNumber()

order = Order()
order.quantity = 5  # 正常
# order.quantity = -1  # 抛出ValueError
```



## 40. 元类（Metaclass）的作用是什么？举例说明其实际应用。
详细答案  
- 元类：  
  - 类的类，控制类的创建过程（如验证属性、自动注册子类）。  
  - 默认元类为 `type`，可自定义（如继承 `type`）。  

- 应用场景：  
  1. ORM框架（如Django模型的字段验证）。  
  2. API接口的自动路由注册。  

代码示例  
```python
class ModelMeta(type):
    def __new__(cls, name, bases, attrs):
        # 自动收集字段名
        fields = [k for k, v in attrs.items() if isinstance(v, Field)]
        attrs['_fields'] = fields
        return super().__new__(cls, name, bases, attrs)

class Field:
    pass

class User(metaclass=ModelMeta):
    name = Field()
    age = Field()

print(User._fields)  # ['name', 'age']
```

