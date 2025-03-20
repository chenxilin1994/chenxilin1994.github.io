---
layout: doc
title: Python文件操作超详细教程
editLink: true
---

# Python文件操作从入门到精通

## 第一章：文件操作就像日常记事（基础概念）

### 1.1 文件是什么？
就像我们平时用笔记本记录事情一样，文件是计算机存储信息的"笔记本"。每个文件都有：
- **名字**：比如 "日记.txt"（后缀表示类型）
- **存放位置**：就像笔记本放在书桌的哪个抽屉
- **内容**：文字、图片、数据等

### 1.2 文件路径详解
假设你的文件结构如下：
```
C:
├─ 用户
│  └─ 小明
│     ├─ 图片
│     │  └─ 假期.jpg
│     └─ 文档
│        └─ 报告.docx
```

**路径类型对比**：
```python
# 相对路径：从当前所在位置出发（假设当前在"小明"文件夹）
'文档/报告.docx'          # 正确 ✔️
'图片/假期.jpg'          # 正确 ✔️

# 绝对路径：从最顶层开始描述
r'C:\用户\小明\文档\报告.docx'  # Windows写法 ✔️
'/home/小明/文档/报告.docx'    # Linux/Mac写法 ✔️

# 使用现代路径工具（推荐）
from pathlib import Path

# 就像拼乐高一样组合路径
doc_path = Path('用户') / '小明' / '文档' / '报告.docx'
print(doc_path)  # 输出：用户\小明\文档\报告.docx
```

## 第二章：读写文件就像记笔记（基础操作）

### 2.1 新建笔记本写日记
```python
# 打开文件准备写（'w'模式就像拿新本子）
with open('我的日记.txt', 'w', encoding='utf-8') as diary:
    diary.write("2023年7月20日 晴\n")
    diary.write("今天学会了用Python写文件！\n")
    # \n表示换行，就像写完一行按回车键

# 查看文件内容（现在去文件夹里双击打开看看吧）
```

### 2.2 继续写日记不覆盖之前内容
```python
with open('我的日记.txt', 'a', encoding='utf-8') as diary:  # 'a'是追加模式
    diary.write("\n晚上补充：月亮好圆啊🌕\n")
```

### 2.3 读取日记内容
```python
# 方法1：一次性读完（适合小文件）
with open('我的日记.txt', 'r', encoding='utf-8') as diary:
    all_content = diary.read()
    print("=== 完整日记 ===")
    print(all_content)

# 方法2：逐行读取（适合大文件）
with open('我的日记.txt', encoding='utf-8') as diary:
    print("\n=== 按行阅读 ===")
    for line_number, line_content in enumerate(diary, 1):
        print(f"第{line_number}行：{line_content.strip()}")
```

输出效果：
```
=== 完整日记 ===
2023年7月20日 晴
今天学会了用Python写文件！

晚上补充：月亮好圆啊🌕

=== 按行阅读 ===
第1行：2023年7月20日 晴
第2行：今天学会了用Python写文件！
第3行：
第4行：晚上补充：月亮好圆啊🌕
```

## 第三章：特殊文件处理（图片/Excel等）

### 3.1 复制照片（二进制文件）
```python
def copy_photo(original_path, new_path):
    try:
        # 'rb'表示用二进制模式读取
        with open(original_path, 'rb') as origin:
            photo_data = origin.read()  # 读取原始照片数据
            
        # 'wb'表示用二进制模式写入
        with open(new_path, 'wb') as copy:
            copy.write(photo_data)  # 写入新文件
            
        print(f"照片已保存到：{new_path}")
        
    except FileNotFoundError:
        print("⚠️ 找不到原始照片，请检查路径")
    except PermissionError:
        print("⛔ 没有权限，请检查文件夹是否可写")

# 使用示例（假设有一张test.jpg）
copy_photo('test.jpg', '备份照片.jpg')
```

### 3.2 处理Excel成绩单（需安装openpyxl）
```python
from openpyxl import Workbook

# 创建新Excel文件
grade_book = Workbook()  # 得到一个空本子
sheet = grade_book.active  # 打开第一页

# 添加表头（就像写标题）
sheet.append(["姓名", "语文", "数学", "英语"])

# 添加数据
sheet.append(["张三", 90, 85, 92])
sheet.append(["李四", 88, 95, 89])

# 保存文件（就像合上本子放进书包）
grade_book.save("期末成绩.xlsx")
print("成绩单已生成！")

# 读取Excel文件
from openpyxl import load_workbook

wb = load_workbook('期末成绩.xlsx')
sheet = wb.active

print("\n=== 成绩列表 ===")
for row in sheet.iter_rows(values_only=True):  # values_only获取实际值
    print(row)
```

输出：
```
成绩单已生成！

=== 成绩列表 ===
('姓名', '语文', '数学', '英语')
('张三', 90, 85, 92)
('李四', 88, 95, 89)
```

## 第四章：高效管理文件（实用技巧）

### 4.1 自动整理照片
```python
from pathlib import Path
import shutil

photo_folder = Path("手机照片")
organized_folder = Path("整理后的照片")

# 创建年份文件夹
for year in ["2021", "2022", "2023"]:
    (organized_folder / year).mkdir(parents=True, exist_ok=True)

# 移动文件（假设文件名包含日期）
for photo in photo_folder.glob("IMG_*.jpg"):
    # 从文件名提取日期（假设格式为IMG_20230720_123456.jpg）
    date_part = photo.name.split('_')[1]
    year = date_part[:4]
    
    target_folder = organized_folder / year
    shutil.move(str(photo), str(target_folder / photo.name))
    print(f"已移动 {photo.name} 到 {year} 文件夹")
```

### 4.2 处理超大文件（避免卡死）
```python
def process_big_file(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as big_file:
        with open(output_file, 'w', encoding='utf-8') as result_file:
            # 逐行处理，内存友好
            for line in big_file:
                processed_line = line.strip().upper()  # 转大写
                result_file.write(processed_line + '\n')
                
    print(f"已完成处理，结果保存到 {output_file}")

# 使用示例（处理100MB的日志文件）
process_big_file("huge_log.txt", "processed_log.txt")
```

## 第五章：常见问题排雷指南

### 5.1 中文乱码问题
```python
try:
    with open('旧文件.txt', 'r') as f:
        content = f.read()
except UnicodeDecodeError:
    print("检测到编码问题，尝试常见中文编码...")
    for encoding in ['gbk', 'utf-16', 'big5']:
        try:
            with open('旧文件.txt', 'r', encoding=encoding) as f:
                content = f.read()
            print(f"使用 {encoding} 编码成功读取！")
            break
        except:
            continue
```

### 5.2 安全删除文件
```python
import send2trash  # 需安装：pip install send2trash

# 比直接删除更安全，会进回收站
send2trash.send2trash('要删除的文件.txt')
print("文件已安全移至回收站")
```

## 实战演练：个人账本程序
```python
from pathlib import Path
import csv
from datetime import datetime

class AccountBook:
    def __init__(self):
        self.file_path = Path("家庭账本.csv")
        self._init_file()
    
    def _init_file(self):
        if not self.file_path.exists():
            with open(self.file_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["日期", "类型", "金额", "说明"])
    
    def add_record(self, type_, amount, note):
        date_str = datetime.now().strftime("%Y-%m-%d %H:%M")
        with open(self.file_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([date_str, type_, amount, note])
        print("√ 记录已保存")
    
    def show_summary(self):
        total_income = 0
        total_expense = 0
        
        with open(self.file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                amount = float(row['金额'])
                if row['类型'] == '收入':
                    total_income += amount
                else:
                    total_expense += amount
                    
        print(f"\n当前结余：{total_income - total_expense:.2f}元")
        print(f"总收入：{total_income:.2f}元")
        print(f"总支出：{total_expense:.2f}元")

# 使用示例
book = AccountBook()
book.add_record("收入", 5000, "工资")
book.add_record("支出", 98.5, "超市购物")
book.show_summary()
```

### 配套练习套餐

1. **新手任务**  
   📌 创建一个"读书笔记.txt"，记录最近读的3本书  
   📌 编写程序统计文件中包含"推荐"一词的行数

2. **进阶挑战**  
   🚀 把手机里的照片按月份整理到不同文件夹  
   🚀 实现CSV转Excel的小工具

