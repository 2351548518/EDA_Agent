"""
路径配置模块 - 定义项目中的重要目录路径

主要路径：
- BACKEND_DIR: backend 目录本身
- PROJECT_ROOT: 项目根目录（backend 的父目录）
- DATA_DIR: 数据目录（PROJECT_ROOT/data）
- FRONTEND_DIR: 前端目录（PROJECT_ROOT/frontend）
"""

from pathlib import Path

# backend 目录的父目录就是项目根目录
BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_DIR.parent

# 数据目录：用于存储上传的文档等
DATA_DIR = PROJECT_ROOT / "data"

# 前端目录：包含静态文件（HTML/CSS/JS）
FRONTEND_DIR = PROJECT_ROOT / "frontend"