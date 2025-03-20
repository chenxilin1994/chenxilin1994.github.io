---
layout: doc
title: Python Web开发入门指南
editLink: true
---

# Python Web开发入门指南

## 理解Web开发就像开餐厅（基础概念）

### 1.1 餐厅运营比喻
- **顾客（客户端）**：使用浏览器访问网站
- **服务员（Web服务器）**：接收请求并传递到后厨
- **厨师（业务逻辑）**：处理具体请求（Python代码）
- **食材（数据库）**：存储需要使用的数据
- **菜单（路由）**：决定不同URL对应的功能

### 1.2 基础技术栈选择
```python
# 微型餐厅（简单应用）
from flask import Flask
app = Flask(__name__)

# 大型餐厅（复杂应用）
# 使用Django等全功能框架
```

## 使用Flask搭建第一个网站

### 2.1 基础网页服务
```python
from flask import Flask

app = Flask(__name__)  # 创建餐厅

@app.route('/')        # 菜单首页
def home():
    return "欢迎来到Python餐厅！"

@app.route('/menu')    # 菜单页面
def show_menu():
    return "<h1>今日特色</h1><ul><li>Python披萨</li><li>Flask面条</li></ul>"

if __name__ == '__main__':
    app.run(debug=True)  # 启动餐厅营业
```

运行后访问：
- `http://localhost:5000/`      查看首页
- `http://localhost:5000/menu`  查看菜单

### 2.2 动态页面示例
```python
from datetime import datetime

@app.route('/greet/<name>')
def greet(name):
    hour = datetime.now().hour
    period = "上午" if hour < 12 else "下午"
    return f'''
    <h2>{name}，{period}好！</h2>
    <p>当前时间：{datetime.now().strftime("%H:%M")}</p>
    <style>
        body {{ background: #f0f0f0; padding: 2rem; }}
    </style>
    '''
```

访问示例：
`http://localhost:5000/greet/张三`

## 制作专业菜单（使用模板引擎）

### 3.1 创建模板文件
新建 `templates/menu.html`：
```html
<!DOCTYPE html>
<html>
<head>
    <title>{{ restaurant }}菜单</title>
</head>
<body>
    <h1>{{ restaurant }}今日菜单</h1>
    <ul>
        {% for item in menu_items %}
        <li>{{ item.name }} - {{ item.price }}元</li>
        {% endfor %}
    </ul>
</body>
</html>
```

### 3.2 渲染模板
```python
from flask import render_template

menu_data = [
    {'name': 'Python披萨', 'price': 68},
    {'name': 'Django牛排', 'price': 98}
]

@app.route('/vip_menu')
def vip_menu():
    return render_template('menu.html',
                         restaurant="高级VIP餐厅",
                         menu_items=menu_data)
```

## 处理顾客点单（表单处理）

### 4.1 创建点单表单
新建 `templates/order.html`：
```html
<form method="POST">
    <input type="text" name="dish" placeholder="菜品名称" required>
    <input type="number" name="quantity" min="1" value="1">
    <button type="submit">提交订单</button>
</form>
```

### 4.2 处理表单数据
```python
from flask import request

orders = []  # 临时存储订单

@app.route('/order', methods=['GET', 'POST'])
def handle_order():
    if request.method == 'POST':
        dish = request.form['dish']
        quantity = request.form['quantity']
        orders.append({'dish': dish, 'quantity': quantity})
        return f"已收到{quantity}份{dish}的订单！"
    return render_template('order.html')
```

## 连接数据库（保存订单）

### 5.1 使用SQLite存储
```python
import sqlite3
from contextlib import closing

def init_db():
    with closing(sqlite3.connect('orders.db')) as conn:
        conn.execute('''CREATE TABLE IF NOT EXISTS orders
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      dish TEXT,
                      quantity INTEGER,
                      time TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')

@app.route('/real_order', methods=['POST'])
def real_order():
    dish = request.form['dish']
    quantity = request.form['quantity']
    
    with closing(sqlite3.connect('orders.db')) as conn:
        conn.execute("INSERT INTO orders (dish, quantity) VALUES (?, ?)",
                    (dish, quantity))
        conn.commit()
    
    return "订单已永久保存！"
```

## 餐厅装修（添加静态资源）

### 6.1 添加CSS样式
新建 `static/style.css`：
```css
body {
    max-width: 800px;
    margin: 0 auto;
    padding: 20px;
    font-family: Arial, sans-serif;
}

.menu-item {
    padding: 10px;
    margin: 5px;
    background: #fff;
    border-radius: 5px;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
}
```

### 6.2 在模板中引用
```html
<head>
    <link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">
</head>
```

## 部署到云厨房（生产环境部署）

### 7.1 使用Gunicorn
```bash
pip install gunicorn
gunicorn -w 4 app:app
```

### 7.2 配置Nginx
```nginx
server {
    listen 80;
    server_name yourdomain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## 常见问题解决方案

### Q1：页面无法加载
```python
# 检查路由是否正确定义
@app.route('/contact')  # 确保有定义该路由
def contact():
    return render_template('contact.html')

# 查看终端输出的访问日志
```

### Q2：表单提交失败
```html
<!-- 检查表单method是否正确 -->
<form method="POST">  <!-- 正确 -->
<!-- 而不是 -->
<form method="POST">  <!-- 错误，多余引号 -->
```

### Q3：静态文件404
```python
# 项目结构必须包含static文件夹
project/
├── app.py
├── static/
│   └── style.css
└── templates/
    └── index.html
```

### 配套实战项目

**项目1：个人博客系统**  
- 实现文章发布功能  
- 添加评论系统  
- 支持Markdown格式

**项目2：在线投票系统**  
- 创建投票主题  
- 实时显示投票结果  
- 防止重复投票
