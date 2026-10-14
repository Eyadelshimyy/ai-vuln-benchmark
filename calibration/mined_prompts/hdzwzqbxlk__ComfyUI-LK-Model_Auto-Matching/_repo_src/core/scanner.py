import logging
import os
import hashlib
import json
import time
import folder_paths

logger = logging.getLogger(__name__)

# 定义要扫描的模型类型 (对应 folder_paths 中的 key，对齐 ComfyUI 最新官方规范)
MODEL_TYPES = {
    "checkpoints": "checkpoints",
    "loras": "loras",
    "vae": "vae",
    "controlnet": "controlnet",
    "upscale_models": "upscale_models",
    "embeddings": "embeddings",
    "clip": "clip",
    "unet": "unet",
    "diffusion_models": "diffusion_models",
    "text_encoders": "text_encoders",
    "clip_vision": "clip_vision",
    "style_models": "style_models",
    "diffusers": "diffusers",
    "gligen": "gligen",
    "hypernetworks": "hypernetworks"
}

# 有效模型文件扩展名 (用于过滤非模型文件)
VALID_MODEL_EXTENSIONS = {
    '.safetensors', '.ckpt', '.pt', '.pth', '.bin', 
    '.gguf', '.onnx', '.pkl', '.sft'
}

def is_valid_model_file(filename):
    if not filename: return False
    _, ext = os.path.splitext(filename)
    return ext.lower() in VALID_MODEL_EXTENSIONS

HASH_VERSION = 1  # 索引结构版本，不兼容时升级

class ModelIndex:
    def __init__(self):
        # 索引文件路径 (保存在 envs/，即 core 上级目录下的 envs/)
        self.index_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "envs", "model_index.json")
        self.data = {
            "version": HASH_VERSION,
            "last_scan": 0,
            "models": {} # { unique_hash: { path, filename, type, size, mtime } }
        }
        self.load_index()

    def load_index(self):
        if os.path.exists(self.index_file):
            try:
                with open(self.index_file, "r", encoding="utf-8") as f:
                    saved_data = json.load(f)
                    if saved_data.get("version") == HASH_VERSION:
                        raw_models = saved_data.get("models", {})
                        # [自愈擦除] 清理物理磁盘上已不存在的无效文件条目
                        cleaned_models = {}
                        removed_count = 0
                        for h, info in raw_models.items():
                            p = info.get("path")
                            if p and os.path.exists(p):
                                cleaned_models[h] = info
                            else:
                                removed_count += 1
                        
                        saved_data["models"] = cleaned_models
                        self.data = saved_data
                        
                        # 若清理了已删除的记录，自动写回索引镜像文件
                        if removed_count > 0:
                            logger.info(f"[AutoMatch] Cleaned {removed_count} deleted model entries from index.")
                            self.save_index()
                    else:
                        logger.warning("[AutoMatch] Index version mismatch, rebuilding...")
            except Exception as e:
                logger.exception(f"[AutoMatch] Failed to load index: {e}")

    def save_index(self):
        try:
            os.makedirs(os.path.dirname(self.index_file), exist_ok=True)
            with open(self.index_file, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.exception(f"[AutoMatch] Failed to save index: {e}")

    def calculate_fast_hash(self, filepath):
        """
        计算快速哈希：Size + MTime + First 1MB + Last 1MB (MD5)
        足以区分大部分模型文件，且速度极快
        """
        try:
            stat = os.stat(filepath)
            file_size = stat.st_size
            mtime = stat.st_mtime
            
            # 基础指纹
            fingerprint = f"{file_size}-{mtime}"
            
            # 读取头尾数据进行哈希 (避免仅靠元数据冲突)
            md5 = hashlib.md5()
            md5.update(fingerprint.encode('utf-8'))
            
            with open(filepath, 'rb') as f:
                # Read first 1MB
                chunk = f.read(1024 * 1024)
                md5.update(chunk)
                
                # Check for last 1MB
                if file_size > 1024 * 1024:
                    f.seek(-1024 * 1024, 2)
                    chunk = f.read(1024 * 1024)
                    md5.update(chunk)
            
            return md5.hexdigest()
        except Exception as e:
            logger.exception(f"[AutoMatch] Hash error {filepath}: {e}")
            return None

    def scan_incremental(self):
        """
        执行极速双向路径对齐扫描 (Bi-directional Path Alignment)
        0 重算 Hash 情况下秒级擦除被删模型，保持 100% 动态对齐
        """
        start_time = time.time()
        logger.info("[AutoMatch] Starting fast bi-directional path alignment scan...")
        
        new_or_updated_count = 0
        
        # 1. 内存/磁盘仅读元数据采样 (Walk disk with os.stat)
        disk_files = {}
        for type_key, folder_key in MODEL_TYPES.items():
            try:
                roots = folder_paths.get_folder_paths(folder_key)
                if not roots:
                    continue
                
                for root_path in roots:
                    if not os.path.exists(root_path):
                        continue
                        
                    for root, dirs, files in os.walk(root_path, followlinks=True):
                        # 过滤隐藏目录
                        dirs[:] = [d for d in dirs if not d.startswith('.')]
                        
                        for filename in files:
                            if filename.startswith('.'):
                                continue
                                
                            _, ext = os.path.splitext(filename)
                            if ext.lower() not in VALID_MODEL_EXTENSIONS:
                                continue
                                
                            full_path = os.path.join(root, filename)
                            
                            try:
                                stat = os.stat(full_path)
                                disk_files[full_path] = {
                                    "type": type_key,
                                    "filename": os.path.relpath(full_path, root_path),
                                    "size": stat.st_size,
                                    "mtime": stat.st_mtime
                                }
                            except OSError:
                                continue
            except Exception as e:
                logger.exception(f"[AutoMatch] Error scanning {type_key}: {e}")
                pass

        # 2. 双向路径擦除与对齐
        next_models = {}
        path_to_hash = {}
        # 存储旧索引中的文件元数据，用于快速比对
        old_model_metadata = {}
        for h, info in self.data["models"].items():
            if info.get("path"):
                path_to_hash[info["path"]] = h
                old_model_metadata[h] = {
                    "size": info.get("size"),
                    "mtime": info.get("mtime"),
                    "type": info.get("type"),
                    "filename": info.get("filename")
                }

        # 统计被删除的文件数量 (基于路径)
        old_paths = set(path_to_hash.keys())
        current_disk_paths = set(disk_files.keys())
        removed_paths = old_paths - current_disk_paths
        
        # 用于存储在磁盘上存在但路径不在旧索引中的文件（可能是新增或移动）
        potential_new_or_moved_files = {}
        for path, meta in disk_files.items():
            if path not in old_paths:
                potential_new_or_moved_files[path] = meta

        # 尝试识别移动文件：通过对比 size + mtime
        # 针对 removed_paths 中的每一个旧文件，尝试在 potential_new_or_moved_files 中找到匹配项
        moved_files_map = {}
        # 创建一个反向映射，通过 size-mtime 组合来查找旧索引中的文件哈希
        size_mtime_to_hash = {}
        for h, meta in old_model_metadata.items():
            key = f"{meta['size']}-{meta['mtime']}"
            # 假设 size-mtime 组合是唯一的，或者我们只关心第一次匹配到的
            if key not in size_mtime_to_hash:
                size_mtime_to_hash[key] = h

        files_to_recalculate_hash = {}

        for new_path, new_meta in potential_new_or_moved_files.items():
            # 尝试根据 size-mtime 找到旧的 hash
            key = f"{new_meta['size']}-{new_meta['mtime']}"
            if key in size_mtime_to_hash:
                old_hash_candidate = size_mtime_to_hash[key]
                # 确认这个 hash 对应的旧路径确实被移除了
                if old_hash_candidate in self.data["models"] and self.data["models"][old_hash_candidate]["path"] in removed_paths:
                    # 这是一个移动的文件，重用旧哈希
                    moved_files_map[new_path] = old_hash_candidate
                    # 从 removed_paths 中移除，因为它不是真的被删除了
                    removed_paths.remove(self.data["models"][old_hash_candidate]["path"])
                    # 将这个旧哈希对应的元数据更新到新路径
                    next_models[old_hash_candidate] = {
                        "path": new_path,
                        "filename": new_meta["filename"],
                        "type": new_meta["type"],
                        "size": new_meta["size"],
                        "mtime": new_meta["mtime"],
                        "hash": old_hash_candidate # 重用旧哈希
                    }
                    new_or_updated_count += 1 # 算作更新
                else:
                    files_to_recalculate_hash[new_path] = new_meta
            else:
                files_to_recalculate_hash[new_path] = new_meta
        
        removed_count = len(removed_paths)

        # 3. 处理磁盘当前存在的文件 (只对新文件/修改文件重算 Hash，旧文件 0ms 秒级复用)
        for path, meta in disk_files.items():
            file_hash = None
            
            # Case 1: 路径在旧索引中存在
            if path in path_to_hash:
                old_hash = path_to_hash[path]
                old_info = self.data["models"].get(old_hash)
                
                # 如果这个文件已经被识别为移动文件的新路径，则跳过
                if path in moved_files_map:
                    continue

                # 时间与大小一致 ➡️ 100% 秒级复用原 Hash (0ms, 不读文件)
                if old_info and old_info.get("size") == meta["size"] and abs(old_info.get("mtime", 0) - meta["mtime"]) < 1.0:
                    file_hash = old_hash
                    next_models[file_hash] = old_info
                    continue
                else:
                    # 文件有变动，才计算 Fast Hash
                    file_hash = self.calculate_fast_hash(path)
                    new_or_updated_count += 1
            elif path in moved_files_map: # 如果是移动文件，已经处理过了
                continue
            else:
                # 全新文件，才计算 Fast Hash
                file_hash = self.calculate_fast_hash(path)
                new_or_updated_count += 1
            
            if file_hash:
                next_models[file_hash] = {
                    "path": path,
                    "filename": meta["filename"],
                    "type": meta["type"],
                    "size": meta["size"],
                    "mtime": meta["mtime"],
                    "hash": file_hash
                }

        # 4. 彻底擦除并同步镜像
        self.data["models"] = next_models
        self.data["last_scan"] = time.time()
        self.save_index()
        
        elapsed = time.time() - start_time
        logger.info(f"[AutoMatch] Alignment finished in {elapsed:.3f}s. Total: {len(next_models)}, Added/Updated: {new_or_updated_count}, Removed: {removed_count}")
        return len(next_models)

    def get_all_models(self):
        """返回所有有效的模型信息（运行时校验物理存在性）"""
        valid_models = []
        for info in self.data["models"].values():
            if os.path.exists(info["path"]):
                valid_models.append(info)
        return valid_models

    def find_local_file(self, filename):
        """
        根据文件名查找本地文件的完整路径 (用于 Hash 匹配)
        """
        # 1. 尝试精确文件名匹配
        for info in self.data["models"].values():
            if (info["filename"] == filename or os.path.basename(info["path"]) == filename) and os.path.exists(info["path"]):
                return info["path"]
        return None

class ModelScanner(ModelIndex):
    pass
