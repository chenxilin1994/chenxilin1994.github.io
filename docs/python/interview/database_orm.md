# 数据库与ORM

## 71. 如何在Python中连接SQLite数据库？
详细答案  
- SQLite：轻量级嵌入式数据库，无需服务器，适合开发和测试。  
- 步骤：  
  1. 使用 `sqlite3` 模块。  
  2. 通过 `connect()` 创建连接（若文件不存在则新建）。  
  3. 创建游标执行SQL语句。  

代码示例  
```python {cmd="python3"}
import sqlite3

# 连接数据库（不存在则创建）
conn = sqlite3.connect('test.db')
cursor = conn.cursor()

# 创建表
cursor.execute('''CREATE TABLE IF NOT EXISTS users 
                (id INTEGER PRIMARY KEY, name TEXT, age INTEGER)''')

# 插入数据
cursor.execute("INSERT INTO users (name, age) VALUES (?, ?)", ("Alice", 30))
conn.commit()

# 查询数据
cursor.execute("SELECT * FROM users")
print(cursor.fetchall())  # [(1, 'Alice', 30)]

conn.close()
```



## 72. 如何防止SQL注入攻击？举例说明。
详细答案  
- 风险：用户输入直接拼接到SQL语句中，攻击者可注入恶意代码。  
- 防护方法：  
  - 使用参数化查询（占位符 `?` 或 `%s`）。  
  - 禁止拼接字符串。  

代码示例  
```python {cmd="python3"}
# 错误示例（易受注入攻击）
user_input = "Alice'; DROP TABLE users;--"
cursor.execute(f"SELECT * FROM users WHERE name = '{user_input}'")

# 正确方式（参数化查询）
cursor.execute("SELECT * FROM users WHERE name = ?", (user_input,))
```



## 73. 解释ORM（对象关系映射）的概念及其优势。
详细答案  
- ORM：将数据库表映射为Python类，记录映射为对象，操作对象即操作数据库。  
- 优势：  
  1. 避免直接写SQL，提高开发效率。  
  2. 自动处理数据类型转换。  
  3. 减少SQL注入风险。  
- 常见库：SQLAlchemy、Django ORM。  

代码示例（SQLAlchemy）  
```python {cmd="python3"}
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    name = Column(String)
    age = Column(Integer)

engine = create_engine('sqlite:///test.db')
Base.metadata.create_all(engine)
```



## 74. 如何在Python中使用SQLAlchemy进行事务管理？
详细答案  
- 事务：一组原子性操作，要么全部成功，要么回滚。  
- 步骤：  
  1. 创建Session。  
  2. 使用 `session.begin()` 或上下文管理器管理事务。  

代码示例  
```python {cmd="python3"}
from sqlalchemy.orm import sessionmaker

Session = sessionmaker(bind=engine)
session = Session()

try:
    # 开启事务
    with session.begin():
        user1 = User(name="Bob", age=25)
        user2 = User(name="Charlie", age=35)
        session.add_all([user1, user2])
except Exception as e:
    print("事务回滚:", e)
    session.rollback()
finally:
    session.close()
```



## 75. 如何用Python连接MySQL数据库？
详细答案  
- 步骤：  
  1. 安装驱动：`pip install mysql-connector-python` 或 `pymysql`。  
  2. 使用 `connect()` 配置主机、用户、密码、数据库名。  

代码示例（使用mysql-connector）  
```python {cmd="python3"}
import mysql.connector

conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="password",
    database="test_db"
)

cursor = conn.cursor()
cursor.execute("SELECT VERSION()")
print(cursor.fetchone())  # 输出MySQL版本
conn.close()
```



## 76. 解释数据库连接池的作用及其Python实现。
详细答案  
- 连接池：预先创建多个数据库连接复用，避免频繁创建/关闭连接的开销。  
- 实现库：`SQLAlchemy` 或 `DBUtils`。  

代码示例（SQLAlchemy连接池）  
```python {cmd="python3"}
from sqlalchemy import create_engine

# 配置连接池大小和超时
engine = create_engine(
    'mysql+pymysql://user:password@localhost/db',
    pool_size=5,
    pool_timeout=30
)

# 使用连接
with engine.connect() as conn:
    result = conn.execute("SELECT 1")
    print(result.scalar())  # 1
```



## 77. 如何在Django ORM中执行复杂查询（如聚合、分组）？
详细答案  
- Django ORM：提供高级查询API，支持聚合（`aggregate`）、分组（`annotate`）。  

代码示例  
```python {cmd="python3"}
from django.db.models import Count, Avg

# 按年龄分组统计用户数量
users = User.objects.values('age').annotate(total=Count('id'))

# 计算平均年龄
avg_age = User.objects.aggregate(avg_age=Avg('age'))
```



## 78. 如何用Python操作MongoDB？
详细答案  
- 步骤：  
  1. 安装驱动：`pip install pymongo`。  
  2. 连接MongoDB，操作数据库和集合。  

代码示例  
```python {cmd="python3"}
from pymongo import MongoClient

client = MongoClient('mongodb://localhost:27017/')
db = client['test_db']
collection = db['users']

# 插入文档
collection.insert_one({"name": "Alice", "age": 30})

# 查询文档
docs = collection.find({"age": {"$gt": 25}})
for doc in docs:
    print(doc)
```



## 79. 解释ACID属性在数据库事务中的意义。
详细答案  
- ACID：  
  1. 原子性（Atomicity）：事务全部成功或全部回滚。  
  2. 一致性（Consistency）：事务前后数据库状态合法。  
  3. 隔离性（Isolation）：并发事务互不干扰。  
  4. 持久性（Durability）：事务提交后数据永久保存。  

代码示例（事务原子性）  
```python {cmd="python3"}
# SQLAlchemy中事务回滚
try:
    with session.begin():
        user = User(name="Dave", age=40)
        session.add(user)
        raise Exception("模拟失败")
except:
    print("事务已回滚，数据未插入")
```



## 80. 如何优化数据库查询性能？举例说明。
详细答案  
- 优化方法：  
  1. 索引：为查询频繁的字段创建索引。  
  2. 批量操作：减少单条插入/更新的开销。  
  3. 惰性加载：仅查询所需字段（如Django的 `only()`）。  

代码示例（SQLAlchemy索引）  
```python {cmd="python3"}
from sqlalchemy import Index

# 为User表的name字段创建索引
Index('idx_user_name', User.name)

# 批量插入
session.bulk_insert_mappings(User, [{'name': 'Eve', 'age': 28}, {'name': 'Frank', 'age': 45}])
```
