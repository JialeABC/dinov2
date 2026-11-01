import torch
from collections import OrderedDict

ckpt_path = r"D:/Deeplearning_code/yolov8/dinov2/output/My_study_dinov2_pth/model_final.rank_0.pth"

# 加载 checkpoint
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

# 判断 checkpoint 类型并提取 state_dict
if isinstance(ckpt, OrderedDict):
    state_dict = ckpt
elif isinstance(ckpt, dict) and "state_dict" in ckpt:
    state_dict = ckpt["state_dict"]
elif isinstance(ckpt, dict) and "model" in ckpt:
    state_dict = ckpt["model"]
elif hasattr(ckpt, "state_dict"):
    # 直接保存了整个模型对象
    print("=== 模型结构 ===")
    print(ckpt)
    state_dict = ckpt.state_dict()
else:
    state_dict = ckpt

# 打印所有参数名和形状
print(f"{'参数名':<60} {'形状':<30} {'元素数':>10}")
print("-" * 102)
total_params = 0
for name, param in state_dict.items():
    shape = tuple(param.shape)
    num = param.numel()
    total_params += num
    print(f"{name:<60} {str(shape):<30} {num:>10,}")

print("-" * 102)
print(f"总参数量: {total_params:,}  ({total_params / 1e6:.2f}M)")