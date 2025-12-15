from transformers import (
    AutoModel, AutoTokenizer, AutoModelForTokenClassification,
    CLIPVisionModel, CLIPImageProcessor
)
import os

def download_and_save_models():
    """
    下载并保存模型到本地目录，并生成 pytorch_model.bin：
    1. bert-base-chinese
    2. openai/clip-vit-base-patch32
    3. hfl/chinese-bert-wwm-ext
    """
    models_to_download = {
        # 文本模型
        "bert-base-chinese": "models/bert-base-chinese",
        # 图像模型
        "openai/clip-vit-base-patch32": "models/clip-vit-base-patch32",
        # 中文 NER 模型
        "hfl/chinese-bert-wwm-ext": "models/chinese-bert-wwm-ext"
    }

    os.makedirs("models", exist_ok=True)

    # ===== 下载文本 BERT 模型 =====
    bert_name = "bert-base-chinese"
    bert_dir = models_to_download[bert_name]
    os.makedirs(bert_dir, exist_ok=True)
    print(f"⏳ 正在下载文本模型 {bert_name} ...")
    bert_model = AutoModel.from_pretrained(bert_name)
    bert_tokenizer = AutoTokenizer.from_pretrained(bert_name)
    # 保存 safetensors + pytorch_model.bin
    bert_model.save_pretrained(bert_dir, safe_serialization=False)
    bert_tokenizer.save_pretrained(bert_dir)
    print(f"✅ 文本模型 {bert_name} 已保存到 {bert_dir}")

    # ===== 下载图像 CLIP 模型 =====
    clip_name = "openai/clip-vit-base-patch32"
    clip_dir = models_to_download[clip_name]
    os.makedirs(clip_dir, exist_ok=True)
    print(f"⏳ 正在下载图像模型 {clip_name} ...")
    clip_model = CLIPVisionModel.from_pretrained(clip_name)
    clip_processor = CLIPImageProcessor.from_pretrained(clip_name)
    clip_model.save_pretrained(clip_dir, safe_serialization=False)
    clip_processor.save_pretrained(clip_dir)
    print(f"✅ 图像模型 {clip_name} 已保存到 {clip_dir}")

    # ===== 下载中文 NER 模型 =====
    ner_name = "hfl/chinese-bert-wwm-ext"
    ner_dir = models_to_download[ner_name]
    os.makedirs(ner_dir, exist_ok=True)
    print(f"⏳ 正在下载中文 NER 模型 {ner_name} ...")
    ner_model = AutoModelForTokenClassification.from_pretrained(ner_name)
    ner_tokenizer = AutoTokenizer.from_pretrained(ner_name)
    ner_model.save_pretrained(ner_dir, safe_serialization=False)
    ner_tokenizer.save_pretrained(ner_dir)
    print(f"✅ 中文 NER 模型 {ner_name} 已保存到 {ner_dir}")

    print("🎉 所有模型已下载并生成 pytorch_model.bin，目录结构如下：")
    for k, v in models_to_download.items():
        print(f" - {k} -> {v}")

if __name__ == "__main__":
    download_and_save_models()
