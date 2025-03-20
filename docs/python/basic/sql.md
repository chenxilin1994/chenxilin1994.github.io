---
layout: doc
title: Python数据库操作指南
editLink: true
---

# Python数据库编程

## 第一章：数据库就像电子档案柜（基础概念）

### 1.1 什么是数据库？
就像公司用档案柜管理文件，数据库是专门管理数据的系统。常见类型：
- **SQL数据库**：像整理好的文件柜（MySQL、PostgreSQL）
- **NoSQL数据库**：像便利贴墙（MongoDB、Redis）

### 1.2 核心概念对照表
| 现实比喻       | 数据库术语      | 代码示例                  |
|----------------|-----------------|---------------------------|
| 档案柜         | 数据库(Database)| `mydata.db`               |
| 文件盒         | 表(Table)       | `users` 表                |
| 文件袋         | 行(Row)         | 用户张三的信息记录        |
| 资料项         | 列(Column)      | 姓名、年龄、邮箱等字段    |

## 第二章：使用SQLite（轻量级数据库）

### 2.1 基本操作流程
```python
import sqlite3

# 连接数据库（不存在则创建）
conn = sqlite3.connect('mydata.db')
cursor = conn.cursor()

# 创建用户表
cursor.execute('''CREATE TABLE IF NOT EXISTS users
               (id INTEGER PRIMARY KEY, 
                name TEXT, 
                age INTEGER)''')

# 插入数据
cursor.execute("INSERT INTO users (name, age) VALUES (?, ?)", ('张三', 25))

# 查询数据
cursor.execute("SELECT * FROM users")
print("所有用户：")
for row in cursor.fetchall():
    print(f"ID: {row[0]}, 姓名: {row[1]}, 年龄: {row[2]}")

# 提交并关闭
conn.commit()
conn.close()
```

### 2.2 可视化工具推荐
使用 **DB Browser for SQLite** 查看数据库：
![DB Browser界面](https://sqlitebrowser.org/images/screenshot.png)

## 第三章：操作MySQL（常用数据库）

### 3.1 连接配置
```python
import pymysql

# 就像输入保险箱密码
config = {
    'host': 'localhost',
    'user': 'root',
    'password': 'mypassword',
    'database': 'mydb',
    'charset': 'utf8mb4'
}

conn = pymysql.connect(**config)
```

### 3.2 安全查询示例
```python
def get_user(email):
    try:
        with conn.cursor() as cursor:
            # 使用参数化查询防止SQL注入
            sql = "SELECT * FROM users WHERE email = %s"
            cursor.execute(sql, (email,))
            return cursor.fetchone()
    except pymysql.Error as e:
        print(f"数据库错误：{e}")
    finally:
        conn.close()

# 使用示例
user = get_user('zhangsan@example.com')
```

## 第四章：使用ORM（对象关系映射）

### 4.1 SQLAlchemy示例
```python
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# 创建基类
Base = declarative_base()

# 定义用户模型
class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    name = Column(String(50))
    email = Column(String(100))

# 初始化数据库
engine = create_engine('sqlite:///mydata.db')
Base.metadata.create_all(engine)

# 操作数据库
Session = sessionmaker(bind=engine)
session = Session()

# 添加新用户
new_user = User(name="李四", email="lisi@example.com")
session.add(new_user)
session.commit()

# 查询用户
users = session.query(User).filter_by(name="李四").all()
```

### 4.2 ORM优势对比
| 操作           | 原生SQL                     | ORM写法                     |
|----------------|-----------------------------|-----------------------------|
| 创建表         | `CREATE TABLE...`           | 定义Python类                 |
| 插入数据       | `INSERT INTO...`            | `session.add(obj)`          |
| 查询数据       | `SELECT * FROM...`          | `session.query(User).all()` |
| 更新数据       | `UPDATE...SET...`           | `obj.name = '新名字'`        |

## 第五章：数据库优化技巧

### 5.1 索引优化
```python
# 在经常查询的字段创建索引
cursor.execute("CREATE INDEX idx_name ON users (name)")
```

### 5.2 批量操作
```python
# 低效方式
for i in range(1000):
    cursor.execute("INSERT INTO data VALUES (?)", (i,))

# 高效批量插入
data = [(i,) for i in range(1000)]
cursor.executemany("INSERT INTO data VALUES (?)", data)
```

### 5.3 连接池配置
```python
from sqlalchemy.pool import QueuePool

engine = create_engine(
    'mysql+pymysql://user:pass@localhost/db',
    poolclass=QueuePool,
    pool_size=10,
    max_overflow=5
)
```

## 第六章：实战项目-博客系统

### 6.1 数据库设计
```python
# posts表结构
CREATE TABLE posts (
    id INT PRIMARY KEY AUTO_INCREMENT,
    title VARCHAR(100) NOT NULL,
    content TEXT,
    author_id INT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

# users表结构
CREATE TABLE users (
    id INT PRIMARY KEY AUTO_INCREMENT,
    username VARCHAR(50) UNIQUE,
    password VARCHAR(100)
);
```

### 6.2 核心功能实现
```python
class BlogDB:
    def __init__(self):
        self.conn = sqlite3.connect('blog.db')
    
    def create_user(self, username, password):
        # 密码加密存储
        hashed_pw = hashlib.sha256(password.encode()).hexdigest()
        self.conn.execute("INSERT INTO users VALUES (?,?)", (username, hashed_pw))
    
    def get_post(self, post_id):
        return self.conn.execute(
            "SELECT * FROM posts WHERE id=?", (post_id,)
        ).fetchone()
    
    # 更多方法...
```

## 常见问题解决方案

### Q1：连接超时处理
```python
import time

def safe_connect(max_retries=3):
    for i in range(max_retries):
        try:
            return pymysql.connect(**config)
        except pymysql.OperationalError:
            wait = 2 ** i  # 指数退避
            print(f"连接失败，{wait}秒后重试...")
            time.sleep(wait)
    raise Exception("无法连接数据库")
```

### Q2：数据备份方案
```python
import shutil
from datetime import datetime

def backup_db():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    backup_name = f"backup_{timestamp}.db"
    shutil.copyfile('mydata.db', backup_name)
    print(f"已创建备份：{backup_name}")
```

### Q3：数据库迁移
```bash
# 导出SQLite数据
sqlite3 mydata.db .dump > backup.sql

# 导入到MySQL
mysql -u root -p mydb < backup.sql
```


### 配套练习项目

**项目1：学生成绩管理系统**  
- 使用SQLite存储学生信息  
- 实现成绩统计功能  
- 生成各科成绩报表

**项目2：电商库存系统**  
- MySQL管理商品数据  
- 实现库存预警功能  
- 使用ORM进行数据操作
