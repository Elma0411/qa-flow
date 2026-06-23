# 文件作用：下载和准备本地评价模型文件。
# 关联说明：为 language_models 和本地评价器准备模型文件。

"""
语义相似度模型下载管理脚本
支持下载多种高质量的中文语义相似度模型
"""

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.core.runtime_paths import MODELS_DIR, model_path

# 支持的模型配置（仅保留必需的两个模型）
SUPPORTED_MODELS = {
    # 语义相似度模型（用于Relevance, Coverage, Overlap, Accuracy）
    'bge-m3': {
        'repo_id': 'BAAI/bge-m3',
        'local_dir': model_path('bge-m3'),
        'size': '2.2GB',
        'description': 'BGE-M3多语言模型，用于语义相似度评估',
        'type': 'semantic'
    },

    # 流畅度评估模型（用于Fluency）
    'chinese-bert-wwm': {
        'repo_id': 'hfl/chinese-bert-wwm-ext',
        'local_dir': model_path('chinese_bert_wwm_ext_pytorch'),
        'size': '400MB',
        'description': '中文BERT模型，用于流畅度评估',
        'type': 'fluency'
    }
}

def check_model_exists(model_key):
    """检查指定模型是否已存在且完整"""
    if model_key not in SUPPORTED_MODELS:
        return False
        
    local_dir = SUPPORTED_MODELS[model_key]['local_dir']
    
    if not os.path.exists(local_dir):
        return False
    
    # 检查必要文件是否存在
    required_files = ['config.json']
    existing_files = os.listdir(local_dir)
    
    # 检查是否有模型权重文件
    weight_files = [f for f in existing_files if f.endswith(('.bin', '.safetensors'))]
    
    if any(file in existing_files for file in required_files) and len(weight_files) > 0:
        return True
    
    return False

def download_model(model_key, force_redownload=False):
    """下载指定模型"""
    if model_key not in SUPPORTED_MODELS:
        print(f"❌ 不支持的模型: {model_key}")
        print(f"支持的模型: {list(SUPPORTED_MODELS.keys())}")
        return False
    
    model_config = SUPPORTED_MODELS[model_key]
    repo_id = model_config['repo_id']
    local_dir = model_config['local_dir']
    size = model_config['size']
    description = model_config['description']
    
    print(f"\n📦 准备下载模型: {model_key}")
    print(f"📝 描述: {description}")
    print(f"📊 大小: {size}")
    print(f"🎯 目标目录: {local_dir}")
    
    # 检查是否已存在
    if check_model_exists(model_key) and not force_redownload:
        print(f"✅ 模型已存在，跳过下载")
        return True
    
    try:
        # 创建目录
        os.makedirs(local_dir, exist_ok=True)
        
        print(f"\n🚀 开始下载 {repo_id}...")
        print("⏳ 请耐心等待，大模型下载可能需要较长时间...")
        
        # 方法1: 使用SentenceTransformer下载
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as e1:
            SentenceTransformer = None  # type: ignore[assignment]
            print(f"⚠️ SentenceTransformer不可用: {str(e1)}")

        if SentenceTransformer is not None:
            try:
                model = SentenceTransformer(repo_id)
                model.save(local_dir)
                print(f"✅ 模型下载成功: {local_dir}")

                # 验证模型
                print("🔍 验证模型...")
                test_model = SentenceTransformer(local_dir)
                test_embeddings = test_model.encode(['测试句子', 'test sentence'])
                print(f"✅ 模型验证成功! 嵌入维度: {test_embeddings.shape}")
                return True

            except Exception as e1:
                print(f"⚠️ SentenceTransformer下载失败: {str(e1)}")
                print("🔄 尝试备用下载方法...")

        # 方法2: 使用huggingface_hub
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True
        )
        print(f"✅ 备用方法下载成功: {local_dir}")
        return True
            
    except Exception as e:
        print(f"❌ 下载失败: {str(e)}")
        print(f"\n💡 手动下载方法:")
        print(f"1. 访问: https://huggingface.co/{repo_id}")
        print(f"2. 下载所有文件到: {local_dir}")
        return False

def list_models():
    """列出所有支持的模型及其状态"""
    print("\n📋 QA评估系统所需模型:")
    print(f"📁 模型根目录: {MODELS_DIR}")
    print("=" * 80)

    for key, config in SUPPORTED_MODELS.items():
        status = "✅ 已下载" if check_model_exists(key) else "❌ 未下载"
        model_type = "🧠 语义模型" if config['type'] == 'semantic' else "📝 流畅度模型"
        print(f"{key:20} | {status:8} | {config['size']:8} | {model_type} | {config['description']}")

def download_all_required():
    """下载所有必需的模型"""
    print("\n🎯 下载QA评估系统所需的所有模型")
    print("=" * 60)

    all_success = True

    for model_key, config in SUPPORTED_MODELS.items():
        print(f"\n📦 下载 {model_key} - {config['description']}")
        success = download_model(model_key)
        if not success:
            print(f"❌ {model_key} 下载失败")
            all_success = False

    if all_success:
        print(f"\n🎉 所有模型下载完成！")
    else:
        print(f"\n⚠️ 部分模型下载失败，请检查网络连接")

    return all_success

def main():
    """主函数"""
    print("🤖 语义相似度模型下载管理工具")
    print("=" * 60)
    
    if len(sys.argv) < 2:
        print("📖 使用方法:")
        print("  python model_downloader.py <command> [model_name]")
        print("\n📋 可用命令:")
        print("  list                    - 列出所有模型状态")
        print("  download <model_name>   - 下载指定模型")
        print("  download-all           - 下载所有必需模型")
        print("\n🎯 必需模型:")
        print("  bge-m3           - 语义相似度模型（必需）")
        print("  chinese-bert-wwm - 流畅度评估模型（必需）")
        
        list_models()
        return
    
    command = sys.argv[1].lower()
    
    if command == 'list':
        list_models()
        
    elif command == 'download':
        if len(sys.argv) < 3:
            print("❌ 请指定要下载的模型名称")
            list_models()
            return
            
        model_name = sys.argv[2]
        if model_name == 'all':
            # 下载所有模型
            for model_key in SUPPORTED_MODELS.keys():
                download_model(model_key)
        else:
            download_model(model_name)
            
    elif command == 'download-all':
        print("📦 开始下载所有必需模型...")
        download_all_required()
            
    else:
        print(f"❌ 未知命令: {command}")
        print("💡 使用 'python model_downloader.py' 查看帮助")

if __name__ == "__main__":
    main()
