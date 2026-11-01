import torch

# 输入输出路径（按你的实际路径修改）
input_path = "D:/Deeplearning_code/yolov8/dinov2/output/My_study_dinov2_pth/model_final.rank_0.pth"
output_path = "D:/Deeplearning_code/yolov8/dinov2/output/My_study_dinov2_pth/pretrain_teacher_backbone.pth"

# 加载原始 checkpoint
print(f"正在加载: {input_path}")
ckpt = torch.load(input_path, map_location="cpu")

# 查看顶层 key，确认结构
print(f"\n顶层 key 数量: {len(ckpt)}")
print(f"顶层 key: {list(ckpt.keys())[:10]}")

# 如果 ckpt 本身是个 dict，里面可能还嵌套了一层 'model' 或 'teacher'
# 先判断一下
if "model" in ckpt:
    ckpt = ckpt["model"]
    print(f"\n提取 'model' 后，key 数量: {len(ckpt)}")

# 提取 teacher.backbone.* 并去掉前缀
teacher_backbone = {}
for k, v in ckpt.items():
    if k.startswith("teacher.backbone."):
        new_key = k.replace("teacher.backbone.", "")
        teacher_backbone[new_key] = v

print(f"\n提取了 {len(teacher_backbone)} 个 teacher backbone 参数")
print(f"前 10 个 key: {list(teacher_backbone.keys())[:10]}")

# 验证一下关键参数是否存在
expected_keys = ["cls_token", "pos_embed", "patch_embed.proj.weight", "norm.weight"]
for key in expected_keys:
    if key in teacher_backbone:
        print(f"  ✓ {key}: {teacher_backbone[key].shape}")
    else:
        print(f"  ✗ {key}: 缺失")

# 保存
torch.save(teacher_backbone, output_path)
print(f"\n已保存: {output_path}")