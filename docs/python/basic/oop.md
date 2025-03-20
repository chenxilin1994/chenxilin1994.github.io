---
layout: doc
title: Python面向对象编程
editLink: true
---

# Python面向对象编程入门指南

## 第一章：理解基本概念

### 1.1 什么是类和对象？

**通俗解释**：  
- **类（Class）**：就像制作饼干的模具，定义了饼干的形状和成分  
- **对象（Object）**：用模具做出来的具体饼干，每个饼干可以有不同装饰  

**代码示例**：  
```python
# 定义一个简单的类（模具）
class Cookie:
    # 初始化方法：创建对象时自动调用
    def __init__(self, topping):
        # self代表当前创建的饼干对象
        # topping是装饰配料，每个饼干可以不同
        self.topping = topping  # 给饼干添加装饰属性

    # 定义对象方法（饼干的功能）
    def describe(self):
        print(f"这是一块{self.topping}饼干")

# 创建两个饼干对象cookie1 = Cookie("巧克力豆")  # 使用模具制作第一块饼干
cookie2 = Cookie("糖霜")      # 制作第二块饼干

# 让饼干执行方法
cookie1.describe()  # 输出：这是一块巧克力豆饼干
cookie2.describe()  # 输出：这是一块糖霜饼干
```


## 第二章：类与对象详解

### 2.1 构造方法 __init__

**关键点解释**：  
- `__init__` 方法在创建对象时自动执行  
- `self` 参数代表当前对象实例，必须放在第一个参数位置  
- 通过`self.属性名`定义对象属性  

**带详细注释的代码**：  
```python
class Student:
    # 构造方法：当执行 Student() 时自动调用
    def __init__(self, name, age):
        # 给当前对象添加name属性
        self.name = name  # self.name 是对象属性，name 是传入参数
        # 给当前对象添加age属性
        self.age = age
        # 类属性：所有学生共享的学校名称
        self.school = "阳光中学"  # 所有学生的school属性初始值相同

    def show_info(self):
        print(f"{self.name}，{self.age}岁，就读于{self.school}")

# 创建学生对象
stu1 = Student("小明", 15)
stu2 = Student("小红", 16)

stu1.show_info()  # 输出：小明，15岁，就读于阳光中学
stu2.show_info()  # 输出：小红，16岁，就读于阳光中学
```


## 第三章：继承与多态

### 3.1 基础继承示例

**逐步说明**：  
1. 创建父类（基类）定义通用属性和方法  
2. 子类通过继承获得父类的功能  
3. 子类可以添加新功能或修改继承的功能  

**详细代码**：  
```python
# 父类：交通工具
class Vehicle:
    def __init__(self, brand):
        self.brand = brand  # 品牌属性
    
    # 通用方法
    def start(self):
        print("启动交通工具")

# 子类：汽车（继承自Vehicle）
class Car(Vehicle):
    def __init__(self, brand, wheels):
        # 调用父类的构造方法初始化brand
        super().__init__(brand)  # super()代表父类
        self.wheels = wheels    # 添加子类特有属性
    
    # 重写父类方法（多态）
    def start(self):
        print(f"{self.brand}汽车正在点火启动")
    
    # 子类特有方法
    def open_sunroof(self):
        print("打开天窗")

# 创建实例
my_car = Car("比亚迪", 4)
print(my_car.brand)   # 输出：比亚迪（继承自父类）
my_car.start()        # 输出：比亚迪汽车正在点火启动（重写后的方法）
my_car.open_sunroof() # 输出：打开天窗（子类特有方法）
```


## 第四章：类的高级特性

### 4.1 私有属性与保护属性

**访问控制说明**：  
- **公有属性**：直接通过对象访问（如`obj.attr`）  
- **保护属性**：约定用单下划线开头（如`_attr`），提示不要直接访问  
- **私有属性**：双下划线开头（如`__attr`），无法直接访问  

**代码示例**：  
```python
class BankAccount:
    def __init__(self, password):
        self.balance = 0      # 公有属性
        self._password = password  # 保护属性（约定不直接访问）
        self.__secret_code = "ABC123"  # 私有属性（实际会被重命名为_BankAccount__secret_code）
    
    def check_balance(self, pwd):
        if pwd == self._password:
            return self.balance
        else:
            return "密码错误"

# 使用示例
account = BankAccount("123456")

# 访问公有属性
print(account.balance)  # 正确：0

# 访问保护属性（不推荐但可以）
print(account._password)  # 输出：123456（会有警告提示）

# 访问私有属性（会报错）
# print(account.__secret_code)  # 错误：AttributeError

# 正确访问方式（实际开发中不应这样做）
print(account._BankAccount__secret_code)  # 输出：ABC123
```


## 第五章：综合应用实例

### 5.1 学生管理系统

**功能需求**：  
- 添加学生信息  
- 显示所有学生  
- 根据姓名查找学生  

**完整代码**：  
```python
class StudentManager:
    def __init__(self):
        self.students = []  # 存储所有学生对象的列表
    
    def add_student(self, name, age):
        # 创建学生对象并添加到列表
        new_student = Student(name, age)
        self.students.append(new_student)
        print(f"成功添加学生：{name}")
    
    def show_all(self):
        print("\n当前所有学生：")
        for index, student in enumerate(self.students, 1):
            print(f"{index}. 姓名：{student.name}，年龄：{student.age}")
    
    def find_student(self, name):
        found = [s for s in self.students if s.name == name]
        if found:
            print(f"找到学生{name}，年龄：{found[0].age}")
        else:
            print(f"未找到学生：{name}")

class Student:
    def __init__(self, name, age):
        self.name = name
        self.age = age

# 使用示例
manager = StudentManager()
manager.add_student("张三", 18)
manager.add_student("李四", 19)
manager.show_all()
manager.find_student("张三")
```


## 学习建议

1. **练习项目**：  
   - 图书管理系统（Book类 + Library类）  
   - 简易购物车系统（Product类 + Cart类）  

2. **调试技巧**：  
   - 使用`print(dir(obj))`查看对象属性  
   - 在VSCode中使用调试器逐步执行  

3. **常见错误**：  
   - 忘记写`self`参数  
   - 混淆类属性和实例属性  
   - 错误使用继承中的`super()`  


