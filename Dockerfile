FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 安装项目依赖
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# 创建模型目录
RUN mkdir -p /app/datasets

# 声明数据卷挂载点
VOLUME ["/app/datasets/final_model"]

# 复制项目文件
COPY . /app/

# 暴露端口
EXPOSE 8000

# 设置默认命令
CMD ["python", "main.py"] 
