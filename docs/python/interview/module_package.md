# 模块与包管理

## 61. 如何创建和使用Python的虚拟环境？
详细答案  
- 虚拟环境的作用：隔离项目依赖，避免不同项目间的包版本冲突。  
- 创建方法：  
  1. venv模块（Python 3.3+内置）：  
     ```bash
     python -m venv myenv          # 创建虚拟环境
     source myenv/bin/activate    # 激活（Linux/macOS）
     myenv\Scripts\activate.bat   # 激活（Windows）
     ```
  2. virtualenv（第三方工具，支持Python 2/3）：  
     ```bash
     pip install virtualenv
     virtualenv myenv            # 创建
     ```

- 依赖管理：  
  - 使用 `pip freeze > requirements.txt` 导出依赖。  
  - 通过 `pip install -r requirements.txt` 安装依赖。  

代码示例  
```bash
# 创建并激活虚拟环境
python -m venv project_env
source project_env/bin/activate

# 安装包并导出依赖
pip install requests
pip freeze > requirements.txt

# 退出虚拟环境
deactivate
```



## 62. 解释 `if __name__ == "__main__"` 的作用。
详细答案  
- 作用：  
  - 当模块被直接运行时，`__name__` 为 `"__main__"`，对应的代码块会被执行。  
  - 当模块被导入时，`__name__` 为模块名，代码块不执行。  
- 用途：  
  - 模块的测试代码隔离，避免被导入时执行。  

代码示例  
```python
# my_module.py
def calculate(x):
    return x * 2

if __name__ == "__main__":
    # 直接运行模块时执行以下测试代码
    print(calculate(5))  # 输出10
```



## 63. 如何管理Python项目的依赖？
详细答案  
- 工具：  
  1. requirements.txt：手动记录依赖及版本。  
  2. pipenv：结合 `Pipfile` 和 `Pipfile.lock` 自动管理依赖树和虚拟环境。  
  3. poetry：现代依赖管理工具，支持依赖解析、打包发布。  

代码示例  
```bash
# 使用pipenv
pip install pipenv
pipenv install requests==2.25.1  # 安装指定版本
pipenv lock                      # 生成锁定文件

# 使用poetry
poetry add pandas                # 添加依赖
poetry install                   # 安装所有依赖
```



## 64. 解释Python的模块搜索路径。
详细答案  
- 搜索顺序：  
  1. 当前脚本所在目录。  
  2. 环境变量 `PYTHONPATH` 中的目录。  
  3. 标准库目录（如 `/usr/lib/python3.9`）。  
  4. `.pth` 文件中的路径（位于 `site-packages`）。  

- 查看路径：  
  ```python
  import sys
  print(sys.path)
  ```

- 动态添加路径：  
  ```python
  sys.path.append("/custom/module/path")
  ```



## 65. 如何使用 `os` 和 `sys` 模块进行系统操作？
详细答案  
- `os` 模块：文件系统操作、进程管理。  
  ```python
  import os
  os.mkdir("new_dir")          # 创建目录
  print(os.listdir("."))       # 列出当前目录文件
  os.environ["PATH"]           # 获取环境变量
  ```

- `sys` 模块：解释器交互、命令行参数。  
  ```python
  import sys
  print(sys.argv)              # 命令行参数列表
  sys.exit(1)                  # 退出程序
  sys.version                  # Python版本信息
  ```



## 66. 如何读写JSON文件？
详细答案  
- 方法：  
  - `json.dump(obj, file)`：写入文件。  
  - `json.load(file)`：读取文件。  

代码示例  
```python
import json

data = {"name": "Alice", "age": 30}

# 写入JSON文件
with open("data.json", "w") as f:
    json.dump(data, f, indent=4)

# 读取JSON文件
with open("data.json") as f:
    loaded = json.load(f)
    print(loaded["name"])  # Alice
```



## 67. 解释Python中的时间处理模块（datetime, time）。
详细答案  
- `datetime`：处理日期和时间对象。  
  ```python
  from datetime import datetime, timedelta
  now = datetime.now()
  tomorrow = now + timedelta(days=1)
  print(now.strftime("%Y-%m-%d"))  # 格式化输出
  ```

- `time`：底层时间操作（如时间戳）。  
  ```python
  import time
  timestamp = time.time()       # 当前时间戳
  time.sleep(1)                 # 休眠1秒
  ```



## 68. 如何进行HTTP请求（标准库与第三方库）？
详细答案  
- 标准库（urllib）：  
  ```python
  from urllib.request import urlopen
  response = urlopen("http://example.com")
  print(response.read().decode())
  ```

- 第三方库（requests）：  
  ```python
  import requests
  r = requests.get("http://example.com")
  print(r.status_code, r.text)
  ```



## 69. 正则表达式在Python中的应用？
详细答案  
- `re` 模块：提供正则表达式操作。  
  ```python
  import re
  text = "电话：123-4567-8900"
  pattern = r"\d{3}-\d{4}-\d{4}"
  match = re.search(pattern, text)
  if match:
      print(match.group())  # 123-4567-8900
  ```



## 70. 如何进行文件的压缩与解压？
详细答案  
- `zipfile` 模块：  
  ```python
  import zipfile

  # 压缩文件
  with zipfile.ZipFile("archive.zip", "w") as zf:
      zf.write("file.txt")

  # 解压文件
  with zipfile.ZipFile("archive.zip") as zf:
      zf.extractall("extracted")
  ```
