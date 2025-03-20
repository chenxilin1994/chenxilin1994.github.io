---
layout: doc
title: Python正则表达式完全指南
editLink: true
---

# Python正则表达式：从模式匹配到文本挖掘

## 正则表达式就像超级搜索（基础概念）

### 1.1 什么是正则表达式？
想象你在玩"Word Search"游戏时：
- 🔍 普通搜索：查找固定单词 "cat"
- 🦸 正则搜索：查找所有以c开头、t结尾的3字母单词（如cat/cot/cut）

正则表达式就是定义这种搜索模式的特殊语法，常用于：
- 验证手机号/邮箱格式
- 批量提取网页数据
- 日志文件分析
- 文本内容替换

### 1.2 快速体验
```python
import re

text = "联系电话：138-1234-5678，备用电话：150 5555 8888"

# 查找所有手机号
pattern = r'\d{3}[- ]?\d{4}[- ]?\d{4}'
phones = re.findall(pattern, text)

print(phones)  # 输出：['138-1234-5678', '150 5555 8888']
```

## 第二章：基础语法就像搭积木（模式构建）

### 2.1 字符匹配
| 模式   | 匹配内容       | 示例            | 匹配结果       |
|--------|----------------|-----------------|----------------|
| `\d`   | 数字字符       | `\d\d`          | "23"           |
| `\w`   | 单词字符       | `\w\w\w`        | "Cat"          |
| `\s`   | 空白字符       | `\d\s\d`        | "1 2"          |
| `.`    | 任意字符       | `c.t`           | "cat", "cbt"   |
| `[...]`| 字符集合       | `[aeiou]`       | "a", "e"等元音 |

### 2.2 量词控制
| 量词   | 作用           | 示例          | 匹配结果       |
|--------|----------------|---------------|----------------|
| `*`    | 0次或多次      | `a*`          | "", "a", "aaa"|
| `+`    | 1次或多次      | `\d+`         | "1", "123"     |
| `?`    | 0次或1次       | `colou?r`     | "color", "colour"|
| `{n}`  | 精确n次        | `\d{4}`       | "2023"         |
| `{n,m}`| n到m次         | `\w{3,5}`     | "abc", "abcde" |

### 2.3 位置锚定
| 锚点   | 作用           | 示例          | 匹配位置       |
|--------|----------------|---------------|----------------|
| `^`    | 字符串开始     | `^Start`      | 开头           |
| `$`    | 字符串结束     | `end$`        | 结尾           |
| `\b`   | 单词边界       | `\bcat\b`     | 匹配"cat"单词  |

## 第三章：实战模式构建（常见场景）

### 3.1 邮箱验证
```python
def validate_email(email):
    pattern = r'^[\w\.-]+@[\w\.-]+\.\w{2,6}$'
    return re.match(pattern, email) is not None

print(validate_email("user@example.com"))  # True
print(validate_email("invalid.email@"))    # False
```

### 3.2 提取HTML内容
```python
html = '<a href="https://example.com">Example</a>'

# 提取链接和文本
pattern = r'<a href="(.*?)">(.*?)</a>'
match = re.search(pattern, html)

if match:
    url = match.group(1)    # https://example.com
    text = match.group(2)   # Example
```

### 3.3 日期格式转换
```python
date_str = "2023-07-20 20:45"

# 从"YYYY-MM-DD"提取日期
match = re.search(r'(\d{4})-(\d{2})-(\d{2})', date_str)
if match:
    year, month, day = match.groups()
    new_format = f"{day}/{month}/{year}"  # 20/07/2023
```

## 第四章：高级技巧与优化

### 4.1 分组捕获
```python
text = "John: 32, Alice: 28"

# 命名分组（?P<name>...）
pattern = r'(?P<name>\w+):\s*(?P<age>\d+)'

for match in re.finditer(pattern, text):
    print(f"{match.group('name')} 年龄是 {match.group('age')}")
```

### 4.2 非贪婪匹配
```python
html = '<div>Content1</div><div>Content2</div>'

# 贪婪模式（默认）
re.findall(r'<div>(.*)</div>', html)  # ['Content1</div><div>Content2']

# 非贪婪模式（加?）
re.findall(r'<div>(.*?)</div>', html)  # ['Content1', 'Content2']
```

### 4.3 预编译正则表达式
```python
# 对于频繁使用的模式
phone_pattern = re.compile(r'\d{3}-\d{4}-\d{4}')

# 复用预编译对象
text = "电话：138-1234-5678"
match = phone_pattern.search(text)
```

## 第五章：调试与可视化工具

### 5.1 正则表达式调试器
使用在线工具调试模式：
- [RegExr](https://regexr.com/)
- [Regex101](https://regex101.com/)

### 5.2 可视化正则表达式
使用 `regex-diagram` 库生成流程图：
```python
# 安装：pip install regex-diagram
from regex_diagram import draw

pattern = r'^(\d{3})-(\d{4})-(\d{4})$'
draw(pattern)  # 生成可视化流程图
```

## 第六章：常见陷阱与解决方案

### 6.1 转义字符问题
```python
# 错误示例（忘记转义）
re.search(r'1+1=2', '1+1=2')  # 匹配失败

# 正确做法
re.search(r'1\+1=2', '1+1=2')  # 需要转义+
```

### 6.2 回溯灾难
```python
# 危险模式（可能导致卡死）
pattern = r'(a+)+b'

# 优化方案
pattern = r'a+b'  # 简化模式
```

### 6.3 多行匹配
```python
text = """Line1
Line2
Line3"""

# 默认不匹配换行符
re.findall('^Line\d', text)  # 仅 ['Line1']

# 使用多行模式
re.findall('^Line\d', text, flags=re.M)  # ['Line1', 'Line2', 'Line3']
```

## 实战项目：日志分析系统

### 日志格式示例
```
2023-07-20 14:22:35 [ERROR] ModuleA: Connection timeout
2023-07-20 14:23:10 [INFO] ModuleB: User login successful
```

### 分析脚本
```python
log_pattern = r'''
    (\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2})  # 时间
    \s\[(\w+)\]                                # 日志级别
    \s(\w+):                                   # 模块名称
    \s(.+)                                     # 消息内容
'''

error_count = 0
with open('app.log') as f:
    for line in f:
        match = re.search(log_pattern, line, re.X)
        if match:
            time, level, module, msg = match.groups()
            if level == 'ERROR':
                error_count += 1
                print(f"[错误] {time} {module}: {msg}")

print(f"发现 {error_count} 个错误日志")
```


---

### 配套练习题

1. **基础练习**  
   📌 编写匹配中国手机号的正则（以13/15/18开头）  
   📌 提取HTML中所有图片链接（`<img src="...">`）

2. **进阶挑战**  
   🚀 解析Apache访问日志  
   🚀 实现简易Markdown标题解析器（# Header）
