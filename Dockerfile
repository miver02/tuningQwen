# 构建阶段
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 安装项目依赖
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# 创建数据目录
RUN mkdir -p /app/datasets

# 声明数据卷挂载点
VOLUME ["/app/datasets"]

# 复制项目文件（排除datasets目录）
COPY . /app/

ENV getmodelid_url = 'http://47.251.15.52:8000/crawler/get-model-id'
# 暴露端口
EXPOSE 8000

# 设置默认命令
CMD ["python", "main.py"]
