"""
Teacher-Student Distillation with PreActResNet152 and CustomResNet18
 - Teacher: PreActResNet152
 - Student: CustomResNet18 (with customized ResidualBlock)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms

import pickle
import numpy as np
import os
from PIL import Image
import multiprocessing
import pandas as pd
from torch.utils.data import Dataset, DataLoader

###############################################################################
# 1) Teacher Network: PreActResNet
###############################################################################
class PreActBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out

class PreActBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes,
                               kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out

class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(PreActResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64,  num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

# Teacher definition
def PreActResNet152(num_classes=10):
    return PreActResNet(PreActBottleneck, [3,8,36,3], num_classes=num_classes)

###############################################################################
# 2) Student Network: CustomResNet18 
###############################################################################
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1,
                 kernel_size=3, skip_kernel_size=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=kernel_size // 2,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=kernel_size,
                               stride=1,
                               padding=kernel_size // 2,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection: 1x1 if in/out differ or stride != 1
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels,
                                  kernel_size=skip_kernel_size,
                                  stride=stride,
                                  padding=skip_kernel_size // 2,
                                  bias=False)
        else:
            self.skip = nn.Identity()
        
    def forward(self, x):
        residual = self.skip(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

class CustomResNet18(nn.Module):
    def __init__(self, num_classes=10,
                 block=ResidualBlock,
                 layers=[2, 2, 2, 2],
                 channels=[48, 96, 192, 320],
                 kernel_size=3,
                 skip_kernel_size=1,
                 pool_size=4):
        super(CustomResNet18, self).__init__()
        
        self.in_channels = channels[0]
        # Initial conv
        self.conv1 = nn.Conv2d(3, self.in_channels,
                               kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        
        # Residual layers
        self.layer1 = self._make_layer(block, channels[0],
                                      layers[0], stride=1,
                                      kernel_size=kernel_size,
                                      skip_kernel_size=skip_kernel_size)
        self.layer2 = self._make_layer(block, channels[1],
                                      layers[1], stride=2,
                                      kernel_size=kernel_size,
                                      skip_kernel_size=skip_kernel_size)
        self.layer3 = self._make_layer(block, channels[2],
                                      layers[2], stride=2,
                                      kernel_size=kernel_size,
                                      skip_kernel_size=skip_kernel_size)
        self.layer4 = self._make_layer(block, channels[3],
                                      layers[3], stride=2,
                                      kernel_size=kernel_size,
                                      skip_kernel_size=skip_kernel_size)
        
        # Global average pool, then FC
        self.avg_pool = nn.AvgPool2d(kernel_size=pool_size)
        self.fc = nn.Linear(channels[3], num_classes)

    def _make_layer(self, block, out_channels, num_blocks,
                   stride, kernel_size, skip_kernel_size):
        strides = [stride] + [1]*(num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, out_channels,
                                stride=s,
                                kernel_size=kernel_size,
                                skip_kernel_size=skip_kernel_size))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

###############################################################################
# 3) Data Loading
###############################################################################
def unpickle(file):
    with open(file, 'rb') as fo:
        dict_data = pickle.load(fo, encoding='bytes')
    return dict_data

class CIFARDataset(Dataset):
    def __init__(self, data_files, transform=None):
        self.data = []
        self.labels = []
        self.transform = transform

        for file in data_files:
            batch = unpickle(file)
            images = batch[b'data']
            labels = batch[b'labels']
            # Reshape to (N, 3, 32, 32)
            images = images.reshape(-1, 3, 32, 32).astype(np.uint8)
            self.data.append(images)
            self.labels.extend(labels)

        self.data = np.vstack(self.data)      # shape = (50000, 3, 32, 32)
        self.labels = np.array(self.labels)   # shape = (50000,)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.data[idx]  # shape (3, 32, 32)
        label = self.labels[idx]
        image = Image.fromarray(np.transpose(image, (1, 2, 0)))  # (H, W, C)

        if self.transform:
            image = self.transform(image)

        return image, label

###############################################################################
# 4) Data Augmentation / Transforms
###############################################################################
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, 
                           saturation=0.1, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])

###############################################################################
# 5) Distillation Loss and Training
###############################################################################
def distillation_loss(student_logits, teacher_logits, labels, T=3.0, alpha=0.5):
    """
    Combines:
     - Hard label cross-entropy with ground-truth
     - Soft label KL-div from teacher
    """
    ce_loss = F.cross_entropy(student_logits, labels)
    # Soft teacher
    student_log_probs = F.log_softmax(student_logits / T, dim=1)
    teacher_probs     = F.softmax(teacher_logits / T, dim=1)
    kl_div = F.kl_div(student_log_probs, teacher_probs,
                      reduction='batchmean') * (T * T)

    loss = alpha * ce_loss + (1 - alpha) * kl_div
    return loss

def train_student_distillation(
    teacher, student, train_loader, test_loader, 
    device, epochs=200, alpha=0.5, T=3.0, 
    lr=0.1, momentum=0.9, weight_decay=5e-4
):
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    student.to(device)
    teacher.to(device)
    student.train()

    optimizer = optim.SGD(student.parameters(), lr=lr, momentum=momentum,
                          weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        student.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            # Student forward
            student_logits = student(images)
            with torch.no_grad():
                teacher_logits = teacher(images)

            # Distillation loss
            loss = distillation_loss(student_logits, teacher_logits,
                                     labels, T=T, alpha=alpha)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = student_logits.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        scheduler.step()
        train_acc = 100.0 * correct / total
        train_loss = running_loss / total

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Loss: {train_loss:.4f} | "
              f"Train Acc: {train_acc:.2f}%")

        test_acc = test_model(student, test_loader, device)
        print(f"  -> Test Acc: {test_acc:.2f}%")

    return student

def test_model(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    return 100.0 * correct / total

###############################################################################
# 6) Test Dataset 
###############################################################################
class TestCIFARDataset(Dataset):
    def __init__(self, images, transform=None):
        self.images = images
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        # (C,H,W)->(H,W,C) for PIL
        image = Image.fromarray(np.transpose(image, (1, 2, 0)))

        if self.transform:
            image = self.transform(image)
        return image

def run_inference(model, test_loader, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for images in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            predictions.extend(predicted.cpu().numpy())
    return predictions

###############################################################################
# 7) Main Execution
###############################################################################
if __name__ == "__main__":
    # Adjust your data directories as needed
    data_dir = "./deep-learning-spring-2025-project-1/cifar-10-python/cifar-10-batches-py"
    train_files = [os.path.join(data_dir, f"data_batch_{i}") for i in range(1, 6)]
    test_file   = os.path.join(data_dir, "test_batch")

    device = torch.device("cuda") 
    num_workers = min(8, multiprocessing.cpu_count() // 2)
    pin_memory = True

    # Create Datasets/Dataloaders
    train_dataset = CIFARDataset(train_files, transform=transform_train)
    test_dataset  = CIFARDataset([test_file], transform=transform_test)

    train_loader  = DataLoader(train_dataset, batch_size=128, shuffle=True,
                               num_workers=num_workers, pin_memory=pin_memory)
    test_loader   = DataLoader(test_dataset, batch_size=128, shuffle=False,
                               num_workers=num_workers, pin_memory=pin_memory)

    # 7.1) Load the TEACHER: PreActResNet152
    teacher = PreActResNet152()
    teacher_weights = torch.load("PreActResNet152_2_model.pth")
    teacher.load_state_dict(teacher_weights)
    print("Teacher loaded from PreActResNet152_2_model.pth")

    # 7.2) Create the STUDENT: CustomResNet18
    student = CustomResNet18(
        num_classes=10,
        layers=[2,2,2,2],
        channels=[48,96,192,320],
        kernel_size=3,
        skip_kernel_size=1,
        pool_size=4
    )
    # Count parameters
    total_params = sum(p.numel() for p in student.parameters())
    print(f"Student total parameters: {total_params:,}")

    # 7.3) Distillation training
    student = train_student_distillation(
        teacher=teacher,
        student=student,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        epochs=200,  
        alpha=0.5,
        T=3.0,
        lr=0.1,
        momentum=0.9,
        weight_decay=5e-4
    )

    # Final test accuracy
    final_test_acc = test_model(student, test_loader, device)
    print(f"Final Student Test Accuracy: {final_test_acc:.2f}%")

    # Save the distilled student
    torch.save(student.state_dict(), "student_customresnet18.pth")
    print("Distilled student saved to 'student_customresnet18.pth'")

    # 7.4) Inference on kaggle test set
    custom_test_path = os.path.join(data_dir, "cifar_test_nolabel.pkl")
    test_data_nolabel = unpickle(custom_test_path)

    custom_images = test_data_nolabel[b'data']  # shape: (N, 3072) or (N, H, W, C)
    test_ids = test_data_nolabel[b'ids']

    if custom_images.shape[1] == 3072:
        custom_images = custom_images.reshape(-1, 3, 32, 32)
    custom_images = custom_images.transpose(0, 3, 1, 2)

    custom_test_dataset = TestCIFARDataset(custom_images, transform=transform_test)
    custom_test_loader = DataLoader(custom_test_dataset, batch_size=1,
                                    shuffle=False, num_workers=num_workers,
                                    pin_memory=pin_memory)

    # Run inference with the student
    student_predictions = run_inference(student, custom_test_loader, device)
    submission_df = pd.DataFrame({
        'ID': test_ids,
        'Labels': student_predictions
    })
    submission_df.to_csv("student_submission.csv", index=False)
    print("Student submission saved to 'student_submission.csv'")
