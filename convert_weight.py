import torch

src = r"D:/Deeplearning_code/yolov8/dinov2/output/My_study_dinov2_pth/pretrain_teacher_backbone.pth"

dst = r"D:/Deeplearning_code/yolov8/dinov2/output/My_study_dinov2_pth/dinov2_vits14_converted.pth"

# 加载官方预训练权重
weights = torch.load(src, map_location="cpu", weights_only=False)

# 添加 model 字段
checkpoint = {
    "model": weights
}

# 保存
torch.save(checkpoint, dst)

print("转换成功！")