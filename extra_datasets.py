from dinov2.data.datasets import ImageNet

for split in ImageNet.Split:
    dataset = ImageNet(split=split, root="D:/Deeplearning_code/yolov8/dinov2/datasets/imagenet", extra="D:/Deeplearning_code/yolov8/dinov2/datasets/imagenet")
    dataset.dump_extra()