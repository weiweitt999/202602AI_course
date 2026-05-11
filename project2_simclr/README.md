# AI HW2: SimCLR Self-Supervised Learning on CIFAR-10

This project implements SimCLR self-supervised learning on the CIFAR-10 dataset using a modified ResNet-18 backbone. The learned representation is evaluated by kNN monitor and linear probing, and compared with a supervised ResNet-18 model trained from scratch.

## Main Results

| Method | Accuracy |
|---|---:|
| SimCLR kNN Monitor, k=20 | 65.96% |
| SimCLR + Linear Probing | 68.16% |
| Supervised ResNet-18 | 85.46% |

## Files

| File | Purpose |
|---|---|
| `check_dataset.py` | Check CIFAR-10 loading and display sample images. |
| `check_augmentation.py` | Show two SimCLR augmented views from one image. |
| `simclr_dataset_test.py` | Test SimCLR dataset and DataLoader output shapes. |
| `model_test.py` | Test modified ResNet-18 backbone and projector head. |
| `loss_test.py` | Test NT-Xent loss function. |
| `train_simclr.py` | Train SimCLR model and save loss curve. |
| `knn_monitor.py` | Evaluate the final SimCLR backbone using kNN monitor. |
| `linear_probe.py` | Train a linear classifier on the frozen SimCLR backbone. |
| `supervised_train.py` | Train supervised ResNet-18 from scratch. |

## How to Run

```bash
python train_simclr.py
python knn_monitor.py
python linear_probe.py
python supervised_train.py
