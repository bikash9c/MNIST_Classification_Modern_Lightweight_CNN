# MNIST Digit Classification with Lightweight CNN

## 🎯 Project Overview

This project implements a lightweight Convolutional Neural Network (CNN) for handwritten digit classification on the MNIST dataset. The model is designed to achieve high accuracy while maintaining a minimal parameter count of **under 20,000 parameters**.

## 📊 Dataset Information

- **Dataset**: MNIST Handwritten Digits
- **Training samples**: 60,000 images
- **Test samples**: 10,000 images
- **Image size**: 28×28 grayscale
- **Classes**: 10 (digits 0-9)
- **Train/Test split**: 85.7% / 14.3%

## 🏗️ Model Architecture

### Net Architecture

The model follows a compact CNN design optimized for parameter efficiency:

```
Input (1×28×28) 
    ↓
┌─────────────────────────────────────────────────────────┐
│                      Block 1                           │
├─────────────────────────────────────────────────────────┤
│ Conv2d(1→6, 3×3, pad=1) → BatchNorm → ReLU             │
│ Conv2d(6→10, 3×3, pad=1) → BatchNorm → ReLU            │
│ MaxPool2d(2×2) → Dropout(0.25)                         │
│ Output: 10×14×14                                        │
└─────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────┐
│                      Block 2                           │
├─────────────────────────────────────────────────────────┤
│ Conv2d(10→16, 3×3, pad=1) → BatchNorm → ReLU           │
│ Conv2d(16→18, 3×3, pad=1) → BatchNorm → ReLU           │
│ MaxPool2d(2×2) → Dropout(0.25)                         │
│ Output: 18×7×7                                          │
└─────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────┐
│                      Block 3                           │
├─────────────────────────────────────────────────────────┤
│ Conv2d(18→32, 3×3) → BatchNorm → ReLU    # 7×7 → 5×5   │
│ Conv2d(32→24, 3×3) → BatchNorm → ReLU    # 5×5 → 3×3   │
│ Conv2d(24→10, 3×3)                       # 3×3 → 1×1   │
│ Global Average Pooling                                  │
│ Output: 10                                              │
└─────────────────────────────────────────────────────────┘
    ↓
Log Softmax → Classification (10 classes)
```

### Architecture Details

| Layer | Type | Input → Output | Kernel | Padding | Parameters |
|-------|------|----------------|---------|---------|------------|
| **Block 1** | | | | | |
| conv1 | Conv2d + BN | 1 → 6 | 3×3 | 1 | 60 + 12 = 72 |
| conv2 | Conv2d + BN | 6 → 10 | 3×3 | 1 | 550 + 20 = 570 |
| pool1 | MaxPool2d | 28×28 → 14×14 | 2×2 | 0 | 0 |
| **Block 2** | | | | | |
| conv3 | Conv2d + BN | 10 → 16 | 3×3 | 1 | 1,456 + 32 = 1,488 |
| conv4 | Conv2d + BN | 16 → 18 | 3×3 | 1 | 2,610 + 36 = 2,646 |
| pool2 | MaxPool2d | 14×14 → 7×7 | 2×2 | 0 | 0 |
| **Block 3** | | | | | |
| conv5 | Conv2d + BN | 18 → 32 | 3×3 | 0 | 5,216 + 64 = 5,280 |
| conv6 | Conv2d + BN | 32 → 24 | 3×3 | 0 | 6,936 + 48 = 6,984 |
| conv7 | Conv2d | 24 → 10 | 3×3 | 0 | 2,170 |
| **Total** | | | | | **19,210** |

### Key Design Features

1. **Progressive Channel Expansion**: 1 → 6 → 10 → 16 → 18 → 32 → 24 → 10
2. **BatchNorm Integration**: Applied after each conv layer except the final classification layer
3. **Strategic Padding**: 
   - Blocks 1 & 2: Padding preserves spatial dimensions
   - Block 3: No padding for gradual spatial reduction
4. **Regularization**: 
   - Dropout2d (0.25) applied after pooling layers
   - BatchNorm for internal covariate shift reduction
5. **Global Average Pooling**: Eliminates need for large fully connected layers
6. **Parameter Efficiency**: Total of 19,210 parameters (well under 20k target)

## 🔧 Technical Implementation

### Data Preprocessing
```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean/std
])
```

### Model Configuration
- **Batch Size**: 129
- **Optimizer**: Adam (lr=0.001, weight_decay=1e-4)
- **Loss Function**: Negative Log-Likelihood Loss (with log_softmax output)
- **Device**: CUDA if available, else CPU

### Training Parameters
- **Epochs**: 14
- **Learning Rate**: 0.001 (initial)
- **Scheduler**: StepLR (step_size=5, gamma=0.5)
- **Weight Decay**: 1e-4

## 📈 Training Results

### Training Logs

**Detailed Training Progress:**

```
Epoch: 1
loss=0.08483710139989853 batch_id=468: 100%|██████████| 469/469 [00:16<00:00, 29.27it/s]
Test set: Average loss: 0.0541, Accuracy: 9826/10000 (98.26%)

Epoch: 2
loss=0.021504759788513184 batch_id=468: 100%|██████████| 469/469 [00:16<00:00, 28.21it/s]
Test set: Average loss: 0.0376, Accuracy: 9873/10000 (98.73%)

Epoch: 3
loss=0.005182644352316856 batch_id=468: 100%|██████████| 469/469 [00:16<00:00, 29.16it/s]
Test set: Average loss: 0.0291, Accuracy: 9903/10000 (99.03%)

Epoch: 4
loss=0.07781617343425751 batch_id=468: 100%|██████████| 469/469 [00:17<00:00, 27.05it/s]
Test set: Average loss: 0.0241, Accuracy: 9921/10000 (99.21%)

Epoch: 5
loss=0.07725208252668381 batch_id=468: 100%|██████████| 469/469 [00:16<00:00, 29.28it/s]
Test set: Average loss: 0.0243, Accuracy: 9922/10000 (99.22%)

[LR Scheduler Step: 0.001 → 0.0005]

Epoch: 6
loss=0.006654506549239159 batch_id=468: 100%|██████████| 469/469 [00:16<00:00, 27.96it/s]
Test set: Average loss: 0.0203, Accuracy: 9936/10000 (99.36%)

Epoch: 7
loss=0.00657331058755517 batch_id=468: 100%|██████████| 469/469 [00:16<00:00, 29.08it/s]
Test set: Average loss: 0.0207, Accuracy: 9928/10000 (99.28%)

Epoch: 8
loss=0.005532475654035807 batch_id=468: 100%|██████████| 469/469 [00:16<00:00, 27.59it/s]
Test set: Average loss: 0.0196, Accuracy: 9931/10000 (99.31%)

Epoch: 9
loss=0.009076717309653759 batch_id=468: 100%|██████████| 469/469 [00:16<00:00, 28.98it/s]
Test set: Average loss: 0.0209, Accuracy: 9938/10000 (99.38%)

Epoch: 10
loss=0.02869863249361515 batch_id=468: 100%|██████████| 469/469 [00:16<00:00, 28.02it/s]
Test set: Average loss: 0.0191, Accuracy: 9938/10000 (99.38%)

[LR Scheduler Step: 0.0005 → 0.00025]

Epoch: 11
loss=0.0020156889222562313 batch_id=468: 100%|██████████| 469/469 [00:15<00:00, 29.34it/s]
Test set: Average loss: 0.0176, Accuracy: 9935/10000 (99.35%)

Epoch: 12
loss=0.028360718861222267 batch_id=468: 100%|██████████| 469/469 [00:16<00:00, 28.22it/s]
Test set: Average loss: 0.0177, Accuracy: 9932/10000 (99.32%)

Epoch: 13
loss=0.04223646596074104 batch_id=468: 100%|██████████| 469/469 [00:16<00:00, 28.10it/s]
Test set: Average loss: 0.0180, Accuracy: 9940/10000 (99.40%)

Epoch: 14
loss=0.015097707509994507 batch_id=468: 100%|██████████| 469/469 [00:16<00:00, 28.19it/s]
Test set: Average loss: 0.0170, Accuracy: 9943/10000 (99.43%)
```

### Performance Metrics

| Metric | Value |
|--------|-------|
| **Final Test Accuracy** | 99.43% |
| **Best Test Accuracy** | 99.43% (Epoch 14) |
| **Initial Test Accuracy** | 98.26% (Epoch 1) |
| **Total Parameters** | 19,210 |
| **Model Size** | ~75 KB |
| **Training Time** | ~16 sec/epoch |
| **Total Training Time** | ~3.7 minutes |

### Learning Curves
The model shows excellent convergence characteristics:

**Test Accuracy Progression:**
- Epoch 1: 98.26% → Epoch 5: 99.22% (Rapid initial improvement)
- Epoch 6-10: 99.28-99.38% (Steady improvement after LR reduction)
- Epoch 11-14: 99.32-99.43% (Fine-tuning with lower LR)

**Loss Reduction:**
- Test loss decreased from 0.0541 → 0.0170 (68.6% reduction)
- Training converged quickly with Adam optimizer
- StepLR scheduler helped fine-tune performance in later epochs

**Key Observations:**
- No signs of overfitting (test accuracy continues improving)
- Stable training with consistent ~28-29 it/s processing speed
- LR reductions at epochs 6 and 11 helped achieve final performance boost

## 🎯 Model Analysis

### Receptive Field Analysis
- **Block 1**: RF = 5×5 (after conv2)
- **Block 2**: RF = 14×14 (after conv4, accounting for pooling)
- **Block 3**: RF = Full image coverage (28×28)

### Parameter Distribution
- **Block 1**: 642 parameters (3.3%)
- **Block 2**: 4,134 parameters (21.5%)  
- **Block 3**: 14,434 parameters (75.2%)

The majority of parameters are concentrated in the final classification block, which is typical for efficient CNN designs.

## 🚀 Key Achievements

- ✅ **Under 20k Parameters**: 19,210 parameters (3.9% under target)
- ✅ **Excellent Accuracy**: 99.43% test accuracy
- ✅ **Fast Training**: 3.7 minutes total training time
- ✅ **Stable Convergence**: Adam optimizer with StepLR scheduling
- ✅ **No Overfitting**: Consistent improvement throughout training
- ✅ **Efficient Architecture**: Modern design with BatchNorm and GAP
- ✅ **Robust Performance**: 57 out of 10,000 test samples misclassified

## 📁 Files Structure

```
├── model.py              # Neural network architecture
├── train.py              # Training script
├── test.py               # Evaluation script
├── utils.py              # Utility functions
├── requirements.txt      # Dependencies
└── README.md            # This file
```

## 🔧 Usage

### Requirements
```bash
pip install torch torchvision matplotlib numpy
```

### Training
```bash
python train.py
```

### Testing
```bash
python test.py
```

## 🤝 Assignment Requirements Met

- [x] **Parameter Constraint**: Model has 19,210 parameters (< 20,000) ✅
- [x] **MNIST Dataset**: Successfully implemented for digit classification ✅
- [x] **CNN Architecture**: Multi-block convolutional design ✅
- [x] **BatchNorm**: Integrated for training stability ✅
- [x] **High Performance**: Achieved 99.43% test accuracy ✅
- [x] **Modern Techniques**: GAP, Dropout, proper normalization ✅
- [x] **Fast Training**: Efficient convergence in 14 epochs ✅
- [x] **Documentation**: Comprehensive README with architecture details ✅

## 📝 Notes

The model demonstrates excellent performance on the MNIST dataset with a parameter-efficient architecture. The combination of BatchNorm, strategic dropout, and learning rate scheduling resulted in stable training and high accuracy. The architecture successfully balances model complexity with performance constraints.

---
**Author**: [Your Name]  
**Date**: [Assignment Date]  
**Course**: [Course Code]