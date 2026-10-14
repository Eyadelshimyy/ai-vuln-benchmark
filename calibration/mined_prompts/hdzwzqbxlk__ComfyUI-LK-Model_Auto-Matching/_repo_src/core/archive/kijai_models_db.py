"""
[v3.3.2] Kijai/WanVideo_comfy 精确模型数据库
用于模糊匹配时的精确文件名查找

基于 2026-02-02 抓取的 193 个模型文件
"""

# 所有 Kijai/WanVideo_comfy 模型文件路径
KIJAI_WANVIDEO_MODELS = [
    # ========== InfiniteTalk ==========
    "InfiniteTalk/Wan2_1-InfiniTetalk-Single_fp16.safetensors",
    "InfiniteTalk/Wan2_1-InfiniteTalk-Multi_fp16.safetensors",
    
    # ========== rCM LoRAs (关键!) ==========
    "LoRAs/rCM/Wan_2_1_T2V_14B_480p_rCM_lora_average_rank_83_bf16.safetensors",
    "LoRAs/rCM/Wan_2_1_T2V_14B_480p_rCM_lora_average_rank_148_bf16.safetensors",
    "LoRAs/rCM/Wan_2_1_T2V_14B_720p_rCM_lora_average_rank_94_bf16.safetensors",
    "LoRAs/rCM/Wan_2_1_T2V_1_3B_480p_rCM_lora_average_rank_64_bf16.safetensors",
    "LoRAs/rCM/Wan22-I2V-A14B-HIGH-rCM6_0_lora_rank_64_bf16.safetensors",
    "LoRAs/rCM/Wan22-I2V-A14B-LOW-rCM1_0_lora_rank_64_bf16.safetensors",
    
    # ========== 主模型 (I2V/T2V) ==========
    "Wan2_1-I2V-14B-480P_fp8_e4m3fn.safetensors",
    "Wan2_1-I2V-14B-480P_fp8_e5m2.safetensors",
    "Wan2_1-I2V-14B-720P_fp8_e4m3fn.safetensors",
    "Wan2_1-I2V-14B-720P_fp8_e5m2.safetensors",
    "Wan2_1-T2V-14B_fp8_e4m3fn.safetensors",
    "Wan2_1-T2V-14B_fp8_e5m2.safetensors",
    "Wan2_1-T2V-1_3B_bf16.safetensors",
    "Wan2_1-T2V-1_3B_fp32.safetensors",
    "Wan2_1-T2V-1_3B_fp8_e4m3fn.safetensors",
    
    # ========== Wan 2.2 ==========
    "Wan2_2-I2V-A14B-HIGH_bf16.safetensors",
    "Wan2_2-I2V-A14B-LOW_bf16.safetensors",
    
    # ========== Lightx2v ==========
    "Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank32_bf16.safetensors",
    "Lightx2v/lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank32_bf16.safetensors",
    "Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32.safetensors",
    
    # ========== CausVid/AccVid ==========
    "Wan21_CausVid_14B_T2V_lora_rank32.safetensors",
    "Wan21_CausVid_14B_T2V_lora_rank32_v2.safetensors",
    "Wan21_AccVid_T2V_14B_lora_rank32_fp16.safetensors",
    "Wan21_AccVid_I2V_480P_14B_lora_rank32_fp16.safetensors",
    "Wan2_1-AccVideo-T2V-14B_fp8_e4m3fn.safetensors",
    "Wan2_1-T2V-14B_CausVid_fp8_e4m3fn.safetensors",
    
    # ========== MoviiGen ==========
    "Wan21_T2V_14B_MoviiGen_lora_rank32_fp16.safetensors",
    "Wan2_1-MoviiGen1_1_fp16.safetensors",
    "Wan2_1-MoviiGen1_1_fp8_e4m3fn.safetensors",
    
    # ========== Anisora ==========
    "Wan2_1-Anisora-I2V-480P-14B_fp16.safetensors",
    "Wan2_1-Anisora-I2V-480P-14B_fp8_e4m3fn.safetensors",
    "LoRAs/AniSora/Wan2_2_I2V_AniSora_3_2_HIGH_rank_64_fp16.safetensors",
    
    # ========== Skyreels ==========
    "Skyreels/Wan2_1-SkyReels-V2-I2V-14B-480P_fp8_e4m3fn.safetensors",
    "Skyreels/Wan2_1-SkyReels-V2-T2V-14B-720P_fp8_e4m3fn.safetensors",
    
    # ========== VACE ==========
    "Wan2_1-VACE_module_14B_bf16.safetensors",
    "Wan2_1-VACE_module_14B_fp8_e4m3fn.safetensors",
    
    # ========== VAE ==========
    "Wan2_1_VAE_bf16.safetensors",
    "Wan2_1_VAE_fp32.safetensors",
    "Wan2_2_VAE_bf16.safetensors",
    
    # ========== Phantom ==========
    "Phantom-Wan-14B_fp16.safetensors",
    "Phantom-Wan-14B_fp8_e4m3fn.safetensors",
    
    # ========== Turbo ==========
    "Wan22-Turbo/Wan2_2-TI2V-5B-Turbo_fp16.safetensors",
    "LoRAs/Wan22-Turbo/Wan22_TI2V_5B_Turbo_lora_rank_64_fp16.safetensors",
    
    # ========== Lightning ==========
    "LoRAs/Wan22-Lightning/Wan22_A14B_T2V_HIGH_Lightning_4steps_lora_250928_rank128_fp16.safetensors",
    "LoRAs/Wan22-Lightning/Wan22_A14B_T2V_LOW_Lightning_4steps_lora_250928_rank64_fp16.safetensors",
]

# Civitai 风格命名 → Kijai 实际文件名映射
CIVITAI_TO_KIJAI_MAP = {
    # aniWan 系列 (Civitai 连写风格)
    "aniwan2114bfp8e4m3fn_i2v480pnew": "Wan2_1-I2V-14B-480P_fp8_e4m3fn.safetensors",
    "aniwan2114bfp8e4m3fn": "Wan2_1-I2V-14B-480P_fp8_e4m3fn.safetensors",
    "aniwan21t2v14b": "Wan2_1-T2V-14B_fp8_e4m3fn.safetensors",
    "aniwani2v14b": "Wan2_1-I2V-14B-480P_fp8_e4m3fn.safetensors",
    
    # rCM LoRA (用户可能缺少 480p)
    "wan_2_1_t2v_14b_rcm_lora_average_rank_83": "LoRAs/rCM/Wan_2_1_T2V_14B_480p_rCM_lora_average_rank_83_bf16.safetensors",
    "wan_2_1_t2v_14b_rcm_lora_average_rank_148": "LoRAs/rCM/Wan_2_1_T2V_14B_480p_rCM_lora_average_rank_148_bf16.safetensors",
    "wan_2_1_t2v_14b_720p_rcm": "LoRAs/rCM/Wan_2_1_T2V_14B_720p_rCM_lora_average_rank_94_bf16.safetensors",
    
    # InfiniteTalk
    "infinitetalk_single": "InfiniteTalk/Wan2_1-InfiniTetalk-Single_fp16.safetensors",
    "infinitetalk_multi": "InfiniteTalk/Wan2_1-InfiniteTalk-Multi_fp16.safetensors",
    "wan2_1_infinitetalk": "InfiniteTalk/Wan2_1-InfiniTetalk-Single_fp16.safetensors",
}


def find_best_match_in_kijai(filename: str) -> tuple:
    """
    在 Kijai 数据库中查找最佳匹配
    返回: (matched_path, score) 或 (None, 0)
    """
    try:
        from rapidfuzz import fuzz, process
    except ImportError:
        return (None, 0)
    
    import os
    base = os.path.splitext(filename)[0].lower()
    
    # 1. 先检查 Civitai 映射表
    base_cleaned = base.replace('-', '_').replace('.', '_')
    for key, value in CIVITAI_TO_KIJAI_MAP.items():
        if key in base_cleaned or base_cleaned in key:
            return (value, 0.99)
    
    # 2. RapidFuzz 模糊匹配
    all_basenames = [os.path.splitext(os.path.basename(p))[0].lower() for p in KIJAI_WANVIDEO_MODELS]
    
    # 使用 token_set_ratio 处理词序不同
    result = process.extractOne(base, all_basenames, scorer=fuzz.token_set_ratio)
    if result and result[1] >= 80:
        matched_idx = all_basenames.index(result[0])
        return (KIJAI_WANVIDEO_MODELS[matched_idx], result[1] / 100)
    
    # 使用 partial_ratio 处理缺少部分的情况
    result = process.extractOne(base, all_basenames, scorer=fuzz.partial_ratio)
    if result and result[1] >= 85:
        matched_idx = all_basenames.index(result[0])
        return (KIJAI_WANVIDEO_MODELS[matched_idx], result[1] / 100)
    
    return (None, 0)
