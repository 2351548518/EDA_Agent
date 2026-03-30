from dotenv import load_dotenv
load_dotenv()  # 必须在 langchain 相关模块 import 之前

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import os

import api as api_module

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"


def create_app() -> FastAPI:
    app = FastAPI(title="Cute Cat Bot API")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"], # # 允许的源，开发阶段允许所有源，生产环境需要指定源
        allow_credentials=True, # 允许携带cookie
        allow_methods=["*"], # 允许的请求方法
        allow_headers=["*"], # 允许的请求头
    )


    """
    先通过 await call_next(request) 让请求继续传递到下一个处理器（比如路由函数），拿到响应对象后，判断请求的路径（path）是否为根路径 /，或者以 .html、.js、.css 结尾。如果是这些静态资源或首页，就在响应头中添加如下字段：
        Cache-Control: no-cache, no-store, must-revalidate：禁止浏览器和中间代理缓存响应内容。
        Pragma: no-cache：兼容老旧 HTTP/1.0 客户端，防止缓存。
        Expires: 0：让资源立即过期。
    这样做的目的是确保前端页面和静态文件每次都从服务器获取最新内容，避免因浏览器缓存导致的页面或脚本更新不及时。
    """

    @app.middleware("http")
    async def _no_cache(request, call_next):
        response = await call_next(request)
        path = request.url.path or ""
        if path == "/" or path.endswith((".html", ".js", ".css")):
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response
    
    # 挂载路由/注册路由 模块化路由
    app.include_router(api_module.router)

    # serve frontend static files at root
    # FastAPI 会自动从 FRONTEND_DIR 目录下查找并返回对应的文件，如果请求路径是 /，则返回 FRONTEND_DIR/index.html
    """
    这是在 FastAPI 中挂载静态文件 的标准写法。
    它的作用是：

    把整个 frontend/ 文件夹的内容暴露在网站的根路径 / 下。
    当用户访问 http://localhost:8000/ 时，自动返回 frontend/index.html（因为 html=True）。
    当用户访问 http://localhost:8000/js/app.js 时，自动返回 frontend/js/app.js 文件。
    当用户访问 http://localhost:8000/css/style.css 时，自动返回对应的 css 文件。
    任何在 frontend/ 目录下存在的文件，都可以通过 URL 直接访问。

    简单说：这行代码让你的 FastAPI 同时变成一个前端静态文件服务器，把前后端合并成一个项目运行。
    """
    
    if FRONTEND_DIR.exists():
        app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="static")

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", 8000)))
