# Python测试

## 81. 单元测试基础
问题：请用`unittest`模块编写一个测试用例，验证函数`add(a, b)`的行为是否符合预期（例如正常整数相加、字符串拼接等），并解释断言方法的作用。

```python
# 被测试函数
def add(a, b):
    return a + b
```

答案：
```python
import unittest

class TestAddFunction(unittest.TestCase):
    def test_add_integers(self):
        self.assertEqual(add(2, 3), 5)  # 验证整数相加
    
    def test_add_strings(self):
        self.assertEqual(add("Hello", "World"), "HelloWorld")  # 验证字符串拼接
    
    def test_add_negative_numbers(self):
        self.assertEqual(add(-1, -2), -3)  # 验证负数相加

if __name__ == '__main__':
    unittest.main()
```

解释：
- `unittest.TestCase` 是所有测试用例的基类。
- `self.assertEqual(a, b)` 断言两个值相等。
- 每个测试方法以 `test_` 开头，框架会自动识别并执行。



## 82. Mock对象的使用
问题：用`unittest.mock`模拟一个网络请求函数`fetch_data(url)`，假设该函数会返回JSON数据。编写测试用例，验证当`fetch_data`返回`{"status": "success"}`时，调用`process_response()`的结果正确。

答案：
```python
from unittest import TestCase, mock
from mymodule import process_response

class TestFetchData(TestCase):
    @mock.patch('mymodule.fetch_data')
    def test_process_response_success(self, mock_fetch):
        # 配置模拟返回值
        mock_fetch.return_value = {"status": "success"}
        
        result = process_response()
        self.assertEqual(result, "Operation succeeded")  # 验证处理结果
        mock_fetch.assert_called_once()  # 确保模拟方法被调用了一次

# 假设 mymodule.py 中的 process_response 调用了 fetch_data()
```

解释：
- `@mock.patch` 装饰器用于替换目标函数为Mock对象。
- `assert_called_once()` 验证方法是否被调用一次。



## 83. 测试异常抛出
问题：编写测试用例，验证当输入非整数参数时，函数`divide(a, b)`会抛出`TypeError`异常。

```python
def divide(a, b):
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        raise TypeError("Arguments must be numbers")
    return a / b
```

答案：
```python
import unittest

class TestDivideFunction(unittest.TestCase):
    def test_divide_invalid_input(self):
        # 验证传入字符串时是否抛出 TypeError
        with self.assertRaises(TypeError) as context:
            divide("10", 2)
        self.assertEqual(str(context.exception), "Arguments must be numbers")

if __name__ == '__main__':
    unittest.main()
```

解释：
- `self.assertRaises(异常类)` 用于捕获并验证代码块是否抛出指定异常。
- `context.exception` 可以获取捕获的异常对象。



## 84. 参数化测试（使用 `pytest` ）
问题：使用 `pytest.mark.parametrize` 对以下函数`square(x)`的多种输入（正数、负数、零）编写参数化测试。

```python
def square(x):
    return x  2
```

答案：
```python
import pytest

@pytest.mark.parametrize("input_val, expected", [
    (2, 4),
    (-3, 9),
    (0, 0),
    (5.5, 30.25)
])
def test_square(input_val, expected):
    assert square(input_val) == expected
```

解释：
- `@parametrize` 允许为测试函数提供多组输入和期望输出。
- 参数化减少重复代码，提高测试覆盖率。



## 85. 集成测试场景
问题：假设有一个数据处理流程：`read_file()` → `clean_data()` → `save_to_db()`。请编写一个集成测试，模拟整个流程并验证最终数据库中的记录数。

答案：
```python
from unittest import TestCase, mock
from myapp import process_pipeline

class TestDataPipeline(TestCase):
    @mock.patch('myapp.save_to_db')
    @mock.patch('myapp.clean_data')
    @mock.patch('myapp.read_file')
    def test_full_pipeline(self, mock_read, mock_clean, mock_save):
        mock_read.return_value = "raw,data"  # 模拟读取文件
        mock_clean.return_value = ["clean", "data"]  # 模拟清洗数据
        mock_save.return_value = True  # 模拟保存成功
        
        result = process_pipeline("dummy_path.csv")
        self.assertTrue(result)
        mock_clean.assert_called_with("raw,data")  # 确保清洗函数被正确调用
```

解释：
- 多个 `@mock.patch` 装饰器按从下到上的顺序传递参数。
- 集成测试关注多个组件的协同工作。



## 86. 测试覆盖率工具
问题：如何使用 `coverage.py` 计算测试覆盖率？请写出命令行操作步骤及解释结果文件。

答案：

1. 安装工具：`pip install coverage`
2. 运行测试并收集数据：
   ```bash
   coverage run -m pytest tests/
   coverage html  # 生成HTML报告
   ```
3. 查看报告：
   - 打开 `htmlcov/index.html`，显示每行代码是否被测试覆盖。
   - 识别未覆盖的代码分支，补充测试用例。



## 87. 测试驱动开发（TDD）
问题：根据TDD流程，编写一个测试和函数`is_palindrome(s: str) -> bool`，判断字符串是否为回文。

答案：
```python
# 先写测试
import unittest

class TestPalindrome(unittest.TestCase):
    def test_valid_palindrome(self):
        self.assertTrue(is_palindrome("racecar"))
        self.assertTrue(is_palindrome("madam"))
    
    def test_invalid_palindrome(self):
        self.assertFalse(is_palindrome("hello"))
        self.assertFalse(is_palindrome("python"))

    def test_edge_cases(self):
        self.assertTrue(is_palindrome("a"))  # 单个字符
        self.assertTrue(is_palindrome(""))   # 空字符串

# 后实现函数
def is_palindrome(s):
    return s == s[::-1]

if __name__ == '__main__':
    unittest.main()
```

解释：
- TDD先写测试再实现功能，确保代码符合需求。
- 边界条件测试（空字符串、单字符）很重要。



## 88. 性能测试
问题：如何对Python函数进行性能测试？编写一个基准测试，比较列表推导式`[i for i in range(n)]`和普通循环生成列表的时间差异。

答案：
```python
import timeit

# 列表推导式
def list_comprehension(n):
    return [i for i in range(n)]

# 普通循环
def loop_method(n):
    result = []
    for i in range(n):
        result.append(i)
    return result

# 测试性能
n = 1000000
time_lc = timeit.timeit(lambda: list_comprehension(n), number=100)
time_loop = timeit.timeit(lambda: loop_method(n), number=100)

print(f"列表推导式耗时: {time_lc:.4f}秒")
print(f"普通循环耗时: {time_loop:.4f}秒")
# 输出结果通常显示推导式更快
```

解释：
- `timeit.timeit` 多次执行代码块，返回总耗时。
- 列表推导式通常比普通循环更高效。



## 89. 测试异步代码
问题：用`pytest-asyncio`测试一个异步函数`async_fetch(url: str)`，假设其返回字符串"data"。

答案：
```python
import pytest
from mymodule import async_fetch

@pytest.mark.asyncio
async def test_async_fetch():
    result = await async_fetch("http://example.com")
    assert result == "data"

# 被测试函数实现：
async def async_fetch(url):
    return "data"  # 模拟异步操作（如aiohttp请求）
```

解释：
- `@pytest.mark.asyncio` 声明测试用例为异步函数。
- 使用 `await` 调用异步方法。



## 90. 使用 `tox` 管理多环境测试
问题：如何用 `tox` 配置Python 3.8和3.9环境下的测试？提供`tox.ini`示例文件。

答案：
tox.ini配置：
```ini
[tox]
envlist = py38, py39

[testenv]
deps =
    pytest
    pytest-asyncio
commands = pytest tests/
```

解释：
- `envlist` 定义要测试的Python版本。
- `deps` 安装依赖后运行测试命令。
- 运行 `tox` 将自动构建多环境并执行测试。

