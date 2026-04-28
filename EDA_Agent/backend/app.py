"""
后端应用入口模块。

主要功能：
1. 创建并配置 FastAPI 应用
2. 配置 CORS 中间件（允许跨域访问）
3. 添加缓存控制中间件（防止浏览器缓存静态资源）
4. 注册 API 路由
5. 挂载前端静态文件目录

整体架构：
- FastAPI 应用作为全栈应用，同时提供后端 API 和前端静态文件服务
- 使用 Uvicorn 作为 ASGI 服务器运行应用
"""

# 加载 .env 环境变量文件，确保在导入 langchain 等模块之前完成
# 这是因为 langchain 会在导入时读取一些环境变量
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
import sys
from pathlib import Path

# 获取项目根目录（backend 的父目录）
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# 将项目根目录添加到 sys.path，以便后续可以 import backend.* 模块
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 导入 API 路由模块
from backend.api.routes import router
# 导入前端静态文件目录路径
from backend.common.paths import FRONTEND_DIR


def create_app() -> FastAPI:
    """
    创建并配置 FastAPI 应用的工厂函数。

    配置内容包括：
    1. 应用标题
    2. CORS 中间件（允许跨域请求）
    3. 缓存控制中间件（防止浏览器缓存静态资源）
    4. API 路由注册
    5. 前端静态文件挂载

    Returns:
        配置好的 FastAPI 应用实例
    """
    app = FastAPI(title="Cute Cat Bot API")

    # 配置 CORS（跨域资源共享）中间件
    # 开发阶段允许所有源，生产环境需要根据实际情况限制
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],           # 允许所有来源访问 API
        allow_credentials=True,         # 允许携带 cookie 等凭证信息
        allow_methods=["*"],            # 允许所有 HTTP 方法（GET、POST、PUT 等）
        allow_headers=["*"],           # 允许所有请求头
    )

    """
    缓存控制中间件的工作原理：

    当一个 HTTP 请求到达时，中间件会先调用 await call_next(request) 让请求继续传递到下一个处理器
    （比如路由函数），拿到响应对象后，判断请求的路径（path）是否为根路径 /，
    或者以 .html、.js、.css 结尾。

    如果是这些静态资源或首页，就在响应头中添加如下字段：
        Cache-Control: no-cache, no-store, must-revalidate  → 禁止浏览器和中间代理缓存响应内容
        Pragma: no-cache                                   → 兼容老旧 HTTP/1.0 客户端，防止缓存
        Expires: 0                                         → 让资源立即过期

    这样做的目的是确保前端页面和静态文件每次都从服务器获取最新内容，
    避免因浏览器缓存导致的页面或脚本更新不及时的问题。
    """

    @app.middleware("http")
    async def _no_cache(request, call_next):
        """
        缓存控制中间件函数。

        作用：
        - 对于根路径 / 和静态资源（.html、.js、.css），强制浏览器每次都从服务器获取最新内容
        - 防止浏览器缓存导致的前端更新不及时问题

        Args:
            request: FastAPI 请求对象
            call_next: 传递给下一个处理器的函数

        Returns:
            添加了缓存控制头的响应对象
        """
        response = await call_next(request)
        path = request.url.path or ""

        # 只有根路径和静态资源文件才添加缓存控制头
        if path == "/" or path.endswith((".html", ".js", ".css")):
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"

        return response

    # 挂载/注册 API 路由模块化路由
    # 所有 /sessions/*、/chat、/documents/* 等路由都在 routes.py 中定义
    app.include_router(router)

    # serve frontend static files at root
    # FastAPI 会自动从 FRONTEND_DIR 目录下查找并返回对应的文件
    # 如果请求路径是 /，则返回 FRONTEND_DIR/index.html

    """
    静态文件挂载说明：

    这是在 FastAPI 中挂载静态文件的标准写法。
    它的作用是把整个 frontend/ 文件夹的内容暴露在网站的根路径 / 下：

    - 当用户访问 http://localhost:8000/ 时，自动返回 frontend/index.html（因为 html=True）
    - 当用户访问 http://localhost:8000/js/app.js 时，自动返回 frontend/js/app.js 文件
    - 当用户访问 http://localhost:8000/css/style.css 时，自动返回对应的 css 文件
    - 任何在 frontend/ 目录下存在的文件，都可以通过 URL 直接访问

    简单说：这行代码让你的 FastAPI 同时变成一个前端静态文件服务器，把前后端合并成一个项目运行。
    """

    # 只有当前端目录存在时才挂载静态文件
    if FRONTEND_DIR.exists():
        # directory: 前端文件目录
        # html=True: 当访问根路径时返回 index.html
        # name="static": 给这个挂载点起一个名字
        app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="static")

    return app


# 创建全局应用实例（单例模式）
app = create_app()


# 以下是直接运行此文件时的启动逻辑
if __name__ == "__main__":
    import uvicorn

    # 使用 uvicorn 运行 FastAPI 应用
    # HOST: 绑定地址，0.0.0.0 表示允许所有网络接口访问
    # PORT: 监听端口，从环境变量读取，默认 8000
    uvicorn.run(
        app,
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8000))
    )