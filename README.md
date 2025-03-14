# Deep Learning Spring 2025 - Project 1

This repository contains implementations of various **ResNet architectures** and **Teacher-Student Distillation models** for CIFAR-10 classification. The repository includes Jupyter notebooks, Python scripts, and utility functions to facilitate training and evaluation.

---

## Repository Structure

```
📂 Other Code/
│── 📜 ResNets1.ipynb               # Jupyter Notebook for training ResNet models (Part 1)
│── 📜 ResNets2.ipynb               # Jupyter Notebook for training ResNet models (Part 2)
│── 📜 teacher-student1.py          # First version of teacher-student distillation training
│── 📜 teacher-student2.py          # Second version of teacher-student distillation training
│── 📜 utils.py                     # Utility functions
│── 📜 download_models.sh           # Script to download pretrained models
│── 📜 requirements.txt             # List of dependencies required to run the project              
```

---

## Installation & Setup

Before running the code, install the necessary dependencies and download pretrained models:

```bash
pip install -r requirements.txt
```

Ensure you have **Python 3.8+** installed.

Run the following script to download pretrained models:

```bash
bash download_models.sh
```
This will save all models into the directory. For more information on the pretrained models, see the **[release notes](https://github.com/SJ00425/DL-Project-1-DJT/releases/tag/v1.0)**.

---

## How to Run

### **1. Training ResNet Models**
To train ResNet models on CIFAR-10, run the Jupyter notebooks. You can easily switch between ResNet variants in the notebook by modifying the model = ... line, since all model classes are already instantiated.:

```bash
jupyter notebook ResNets1.ipynb
jupyter notebook ResNets2.ipynb
```

---

### **2. Running Teacher-Student Distillation**
Train a student model using a pretrained teacher:

```bash
python teacher-student1.py
```
or  
```bash
python teacher-student2.py
```



