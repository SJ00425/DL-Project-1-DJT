"""
Teacher-Student Distillation with WideResNet on CIFAR-10
 - Teacher: wide ResNet-40-10
 - Student: wide ResNet-28-3.7 
"""

import os
import pickle
import numpy as np
import pandas as pd
import multiprocessing

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# -------------------------------------------------------------------------
# 1) WideResNet Definitions
# -------------------------------------------------------------------------
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3,
                     stride=stride, padding=1, bias=False)

class WideBasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, dropout_rate=0.0):
        super(WideBasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, out_planes, stride)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.conv2 = conv3x3(out_planes, out_planes, stride=1)

        # If stride != 1 or channel sizes differ, use a 1x1 conv
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1,
                          stride=stride, bias=False)
            )

        self.dropout_rate = dropout_rate

    def forward(self, x):
        out = F.relu(self.bn1(x), inplace=True)
        if self.dropout_rate > 0:
            out = F.dropout(out, p=self.dropout_rate, training=self.training)
        out = self.conv1(out)

        out = F.relu(self.bn2(out), inplace=True)
        if self.dropout_rate > 0:
            out = F.dropout(out, p=self.dropout_rate, training=self.training)
        out = self.conv2(out)

        out += self.shortcut(x)
        return out

class WideResNet(nn.Module):
    def __init__(self, depth, widen_factor, num_classes=10, dropout_rate=0.0):
        super(WideResNet, self).__init__()
        assert (depth - 4) % 6 == 0, "for WideResNet, (depth-4) must be divisible by 6."
        n = (depth - 4) // 6

        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3,
                               stride=1, padding=1, bias=False)

        # Channels after scaling
        ch1 = int(16 * widen_factor)
        ch2 = int(32 * widen_factor)
        ch3 = int(64 * widen_factor)

        # Build stages
        self.layer1 = self._make_layer(ch1, n, stride=1, dropout_rate=dropout_rate)
        self.layer2 = self._make_layer(ch2, n, stride=2, dropout_rate=dropout_rate)
        self.layer3 = self._make_layer(ch3, n, stride=2, dropout_rate=dropout_rate)

        self.bn = nn.BatchNorm2d(ch3)
        self.linear = nn.Linear(ch3, num_classes)

    def _make_layer(self, out_planes, num_blocks, stride, dropout_rate):
        strides = [stride] + [1]*(num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(WideBasicBlock(self.in_planes, out_planes, s, dropout_rate))
            self.in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn(out), inplace=True)
        # global average pool
        out = F.avg_pool2d(out, out.size(2))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def wrn(depth=28, widen_factor=3.7, num_classes=10, dropout_rate=0.0):
    return WideResNet(depth=depth, widen_factor=widen_factor,
                      num_classes=num_classes, dropout_rate=dropout_rate)

# -------------------------------------------------------------------------
# 2) Dataset Handling (CIFAR-10)
# -------------------------------------------------------------------------
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
            images = batch[b'data']     # shape (10000, 3072)
            labels = batch[b'labels']   # list of 10000
            # Reshape to (N, 3, 32, 32)
            images = images.reshape(-1, 3, 32, 32).astype(np.uint8)

            self.data.append(images)
            self.labels.extend(labels)

        self.data = np.vstack(self.data)      # shape (50000, 3, 32, 32)
        self.labels = np.array(self.labels)   # shape (50000,)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.data[idx]  # shape (3, 32, 32)
        label = self.labels[idx]
        # Convert to PIL
        img = np.transpose(img, (1, 2, 0))
        img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)
        return img, label

class TestCIFARDataset(Dataset):
    def __init__(self, images, transform=None):
        self.images = images
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]  # shape (3, 32, 32) or (H, W, C)
        # If shape is (C,H,W), transpose to (H,W,C) for PIL
        if img.shape[0] == 3 and len(img.shape) == 3:
            img = np.transpose(img, (1,2,0))
        img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)
        return img

# -------------------------------------------------------------------------
# 3) Distillation: Hard + Soft Label Loss
# -------------------------------------------------------------------------
def distillation_loss(student_logits, teacher_logits, labels, T=4.0, alpha=0.5):
    # Hard-label CE
    ce_loss = F.cross_entropy(student_logits, labels)

    # Soft teacher
    student_log_probs = F.log_softmax(student_logits / T, dim=1)
    teacher_probs     = F.softmax(teacher_logits / T, dim=1)
    kl_div = F.kl_div(student_log_probs, teacher_probs,
                      reduction='batchmean') * (T * T)

    # Weighted sum
    loss = alpha * ce_loss + (1 - alpha) * kl_div
    return loss

# -------------------------------------------------------------------------
# 4) Distillation Training and Evaluation
# -------------------------------------------------------------------------
def train_student_distillation(
    teacher, student,
    train_loader, test_loader, device,
    epochs=250, alpha=0.5, T=4.0,
    lr=0.1, momentum=0.9, weight_decay=5e-4
):
    # Freeze teacher
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    student.to(device)
    teacher.to(device)
    student.train()

    optimizer = optim.SGD(student.parameters(), lr=lr,
                          momentum=momentum, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        student.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            # Forward
            student_logits = student(images)
            with torch.no_grad():
                teacher_logits = teacher(images)

            # Distillation
            loss = distillation_loss(student_logits, teacher_logits, labels,
                                     T=T, alpha=alpha)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

            # Accuracy
            _, predicted = student_logits.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        scheduler.step()
        train_loss = running_loss / total
        train_acc = 100.0 * correct / total

        # Evaluate on test set
        test_acc = test_model(student, test_loader, device)

        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {train_loss:.4f} "
              f"| Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")

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

def run_inference(model, test_loader, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for images in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            preds.extend(predicted.cpu().numpy())
    return preds

# -------------------------------------------------------------------------
# 5) Main Execution
# -------------------------------------------------------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Paths to CIFAR-10 data
    data_dir = "./deep-learning-spring-2025-project-1/cifar-10-python/cifar-10-batches-py" 
    train_files = [os.path.join(data_dir, f"data_batch_{i}") for i in range(1, 6)]
    test_file   = os.path.join(data_dir, "test_batch")

    # Transforms
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1,
                               saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])

    # Load training/test sets
    train_dataset = CIFARDataset(train_files, transform=transform_train)
    test_dataset  = CIFARDataset([test_file], transform=transform_test)

    # DataLoaders
    num_workers = min(8, multiprocessing.cpu_count() // 2)
    pin_memory = (device.type == 'cuda')

    train_loader = DataLoader(train_dataset, batch_size=128,
                              shuffle=True,  num_workers=num_workers,
                              pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=128,
                             shuffle=False, num_workers=num_workers,
                             pin_memory=pin_memory)

    # ---------------------------------------------------------------------
    # 5.1) Load the TEACHER (Large WRN) from 'wideresnet.pth'
    # ---------------------------------------------------------------------
    teacher = WideResNet(depth=40, widen_factor=10, num_classes=10, dropout_rate=0.3)
    teacher.load_state_dict(torch.load("teacher_wideResNet_40-10.pth"))
    teacher.to(device)
    teacher.eval()

    # ---------------------------------------------------------------------
    # 5.2) Create the STUDENT (Smaller WRN) under 5M parameters
    # ---------------------------------------------------------------------
    student = WideResNet(depth=28, widen_factor=3.7, num_classes=10, dropout_rate=0.0)
    student.to(device)

    total_params = sum(p.numel() for p in student.parameters())
    print(f"Student param count: {total_params:,} (should be < 5M)")

    # ---------------------------------------------------------------------
    # 5.3) Distill
    # ---------------------------------------------------------------------
    distill_epochs = 250
    student = train_student_distillation(
        teacher=teacher,
        student=student,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        epochs=distill_epochs,
        alpha=0.5,
        T=4.0,
        lr=0.1,
        momentum=0.9,
        weight_decay=5e-4
    )

    # Final test accuracy
    final_test_acc = test_model(student, test_loader, device)
    print(f"Final Student Test Accuracy: {final_test_acc:.2f}%")

    # Save the student
    torch.save(student.state_dict(), "student_wrn_under5M.pth")
    print("Distilled student saved to 'student_wrn_under5M.pth'")

    # ---------------------------------------------------------------------
    # 5.4) Inference on Custom No-Label Test (Optional)
    # ---------------------------------------------------------------------
    custom_test_path = os.path.join(data_dir, "cifar_test_nolabel.pkl")
    if os.path.exists(custom_test_path):
        test_nolabel = unpickle(custom_test_path)
        custom_images = test_nolabel[b'data']
        test_ids = test_nolabel[b'ids']

        if custom_images.shape[1] == 3072:
            # Reshape to (N,3,32,32)
            custom_images = custom_images.reshape(-1, 3, 32, 32)
        custom_images = custom_images.transpose(0, 3, 1, 2)

        # Create dataset/dataloader
        custom_transform = transform_test  # same as test
        custom_dataset = TestCIFARDataset(custom_images, transform=custom_transform)
        custom_loader  = DataLoader(custom_dataset, batch_size=1,
                                    shuffle=False, num_workers=num_workers,
                                    pin_memory=pin_memory)

        # Predict
        preds = run_inference(student, custom_loader, device=device)

        # Make CSV
        submission = pd.DataFrame({'ID': test_ids, 'Labels': preds})
        submission.to_csv("student_submission_wrn.csv", index=False)
        print("Custom test submission saved to 'student_submission_wrn.csv'")
