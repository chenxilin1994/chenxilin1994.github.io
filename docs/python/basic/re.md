# Python正则表达式

## 一、正则表达式简介
正则表达式（Regular Expression, 简称Regex）是一种用于**匹配字符串模式**的工具，常用于：
- 验证字符串格式（如邮箱、手机号）
- 提取特定内容（如网页链接、日期）
- 批量替换文本

Python通过 `re` 模块提供正则支持，需先导入：
```python {cmd="python3"}
import re
```


## 二、基础语法

### 1. 普通字符
直接匹配自身，如 `a` 匹配字符 "a"。

### 2. 元字符（特殊符号）
| 元字符 | 说明                     |
|--------|--------------------------|
| `.`    | 匹配**任意单个字符**（除换行符） |
| `^`    | 匹配字符串的**开头**         |
| `$`    | 匹配字符串的**结尾**         |
| `*`    | 前一个字符**0次或多次**       |
| `+`    | 前一个字符**1次或多次**       |
| `?`    | 前一个字符**0次或1次**        |
| `{m,n}`| 前一个字符**m到n次**          |
| `[...]`| 匹配字符集合中的任意一个字符   |
| `\|`     | 逻辑**或**，匹配左右任意表达式 |

**示例**：
- `a.b` 匹配 "aab", "a5b", "a b"
- `^Hello` 匹配以 "Hello" 开头的字符串
- `world$` 匹配以 "world" 结尾的字符串


### 3. 常用元字符缩写
| 缩写 | 说明                   |
|------|------------------------|
| `\d` | 数字（等价于 `[0-9]`）    |
| `\D` | 非数字                 |
| `\s` | 空白字符（空格、制表符等）|
| `\S` | 非空白字符             |
| `\w` | 单词字符（字母、数字、下划线）|
| `\W` | 非单词字符             |



### 4. 量词与贪婪模式
- **贪婪模式**（默认）：尽可能匹配更长的字符串。
- **非贪婪模式**：在量词后加 `?`，匹配最短的字符串。

**示例**：
```python {cmd="python3"}
text = "abbbc"
re.findall("ab+", text)    # 匹配 ['abbb']（贪婪）
re.findall("ab+?", text)   # 匹配 ['ab']（非贪婪）
```


## 三、re模块常用函数

### 1. re.match()
从字符串**开头**匹配模式，成功返回`Match`对象，否则返回`None`。
```python {cmd="python3"}
result = re.match(r'\d+', '123abc')
if result:
    print(result.group())  # 输出 '123'
```

### 2. re.search()
扫描整个字符串，返回**第一个**匹配结果。
```python {cmd="python3"}
result = re.search(r'\d+', 'abc456def')
print(result.group())  # 输出 '456'
```

### 3. re.findall()
返回所有匹配结果的**列表**。
```python {cmd="python3"}
results = re.findall(r'\d+', 'a1b22c333')
print(results)  # 输出 ['1', '22', '333']
```

### 4. re.finditer()
返回所有匹配结果的**迭代器**，适合处理大文本。
```python {cmd="python3"}
for match in re.finditer(r'\d+', 'a1b22c333'):
    print(match.group())
```

### 5. re.sub()
替换匹配的字符串。
```python {cmd="python3"}
text = re.sub(r'\d+', 'X', 'a1b22c333')
print(text)  # 输出 'aXbXcX'
```

### 6. re.split()
按模式分割字符串。
```python {cmd="python3"}
parts = re.split(r'\d+', 'a1b22c333')
print(parts)  # 输出 ['a', 'b', 'c', '']
```


## 四、分组与捕获
用 `()` 分组，可通过 `group(n)` 获取子组。

### 示例：提取日期
```python {cmd="python3"}
text = "2023-10-05"
pattern = r'(\d{4})-(\d{2})-(\d{2})'
match = re.match(pattern, text)
if match:
    year = match.group(1)  # '2023'
    month = match.group(2) # '10'
    day = match.group(3)   # '05'
```

### 非捕获分组
使用 `(?:...)` 避免捕获分组。
```python {cmd="python3"}
pattern = r'(?:\d{4})-\d{2}'  # 不捕获年份
```

## 五、编译正则表达式
预编译正则表达式可提升效率，适用于多次使用同一模式。
```python {cmd="python3"}
pattern = re.compile(r'\d+')
result = pattern.findall('a1b22c333')
```

## 六、高级技巧

### 1. 零宽断言
| 语法       | 说明                     |
|------------|--------------------------|
| `(?=exp)`  | 匹配后面是exp的位置      |
| `(?!exp)`  | 匹配后面不是exp的位置    |
| `(?<=exp)` | 匹配前面是exp的位置      |
| `(?<!exp)` | 匹配前面不是exp的位置    |

**示例**：匹配后面不是数字的字母
```python {cmd="python3"}
re.findall(r'[a-zA-Z]+(?!\d)', 'abc123 def')  # 匹配 ['def']
```

### 2. 条件匹配
```python {cmd="python3"}
pattern = r'(\d{5})(?(1)-\d{4})'  # 如果前5位匹配，则匹配后4位
```

### 3. 添加注释
使用 `(?#注释)` 或 `re.VERBOSE` 模式。
```python {cmd="python3"}
pattern = re.compile(r'''
    \d+  # 匹配数字
    [a-z]+  # 匹配小写字母
''', re.VERBOSE)
```


## 七、常见正则示例

### 1. 邮箱验证
```python {cmd="python3"}
pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
```

### 2. URL提取
```python {cmd="python3"}
pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
```

### 3. 匹配HTML标签
```python {cmd="python3"}
pattern = r'<([a-z]+)([^>]*)>(.*?)</\1>'
```

### 4. 日期格式
```python {cmd="python3"}
pattern = r'\d{4}-\d{2}-\d{2}'
```


## 八、注意事项
1. **转义字符**：在Python字符串中使用 `\` 需写为 `\\`，或使用原始字符串 `r''`。
2. **贪婪陷阱**：避免过度匹配，合理使用 `?` 限制量词。
3. **性能优化**：复杂正则尽量预编译，减少回溯。

