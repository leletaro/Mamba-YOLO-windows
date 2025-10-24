# =============== mytrain.py (clean) ===============
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTHONHASHSEED"] = "0"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # 可选

from ultralytics import YOLO

if __name__ == "__main__":
    # 1) 从自定义 YAML 构建结构，并用 yolov8n.pt 迁移权重
    model_yaml = r"ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T.yaml"
    pretrained = r"yolov8n.pt"
    model = YOLO(model_yaml).load(pretrained)

    # 2) 训练（YOLO 标签格式；你的夜间数据）

    results = model.train(
        data="APDD-DNV_night.yaml",  # 仅修改自己的yaml文件
        imgsz=320,
        batch=1,
        epochs=1,          
        amp=False,
        workers=0,
        val=False,         # 先关闭验证，等确认训练稳定后再打开
        device="0",
        optimizer="SGD",
        cache=False,
        plots=False,
        name="debug_warmup_fast",
    )
