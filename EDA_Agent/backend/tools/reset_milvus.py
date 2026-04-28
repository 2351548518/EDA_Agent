"""
重置 Milvus 集合脚本
当修改了 embedding 模型导致向量维度发生变化时，需要运行此脚本重建集合
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.rag.vector_store.milvus_client import MilvusManager

load_dotenv()

def reset_collection():
    """删除旧集合并创建新集合"""
    manager = MilvusManager()
    
    print(f"正在连接 Milvus: {manager.host}:{manager.port}")
    print(f"集合名: {manager.collection_name}")
    
    # 检查集合是否存在
    if manager.has_collection():
        print("✗ 检测到旧集合，正在删除...")
        manager.drop_collection()
        print("✓ 旧集合已删除")
    else:
        print("✓ 无旧集合，跳过删除")
    
    # 创建新集合（使用 Qwen3-VL-Embedding-8B 的维度 4096）
    print("正在创建新集合（向量维度: 4096）...")
    manager.init_collection(dense_dim=4096)
    print("✓ 新集合创建成功")
    
    print("\n========== 重置完成 ==========")
    print("接下来请重新上传文档以生成新的向量嵌入")

if __name__ == "__main__":
    try:
        reset_collection()
    except Exception as e:
        print(f"✗ 错误: {e}")
        import traceback
        traceback.print_exc()
