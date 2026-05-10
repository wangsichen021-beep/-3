# -3
环境：Python 3.13
PyTorch 2.11.0 + CUDA 12.8
Torchvision 0.26.0
一次性训练三种损失配置：
python train_task3.py --loss all --epochs 12 --batch-size 16 --image-size 128 --base-channels 16 --runs-dir runs/task3_final --amp
使用训练好的 checkpoint 在验证集上测试：
python evaluate_task3.py --checkpoint runs/task3_final/ce_dice/best.pt --data-root data --no-download --batch-size 16 --image-size 128 --base-channels 16 --amp
