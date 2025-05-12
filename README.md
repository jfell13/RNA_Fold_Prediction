# RNA_Fold_Prediction

This project explores machine learning models for predicting RNA 3D structure from sequence, specifically targeting the prediction of the **C1′ atom coordinates**. It is part of a broader effort to understand RNA folding mechanisms using graph-based deep learning and other neural architectures.

## 🔬 Background

RNA structure plays a crucial role in its function, but predicting its 3D conformation from sequence alone remains a challenging problem. Inspired by the [Stanford RNA 3D Folding](https://www.kaggle.com/competitions/stanford-rna-folding) Kaggle competition, this repo implements baseline and advanced models for this task.

## 📁 Project Structure

```
RNA_Fold_Prediction/
│
├── data/                     # Scripts or examples for loading and formatting RNA data
├── models/
│   ├── linear_model/         # Simple linear regression model predicting coordinates
│   └── gcn_model/            # Graph Convolutional Network for structure prediction
│
├── utils/                    # Helper scripts (e.g. RMSD calculation, alignment)
├── train.py                 # Script to train a selected model
├── evaluate.py              # Evaluation and RMSD analysis
├── requirements.txt         # Dependencies
└── README.md
```

## 🧠 Models

### 1. Linear Model
A basic fully connected neural network mapping nucleotide features to C1′ coordinates.

- Input: One-hot encoded nucleotide sequence
- Output: 3D coordinates of each C1′ atom
- Loss: RMSD-based loss with Kabsch alignment

### 2. GCN Model
A Graph Convolutional Network that treats RNA as a graph, where each node is a nucleotide and edges are based on proximity or sequence adjacency.

- Input: Graph representation of RNA
- Architecture: `torch_geometric`-based GCN layers + linear projection
- Loss: Smooth L1 Loss + RMSD monitoring

## ⚙️ Installation

Clone the repository and install dependencies:
```bash
git clone https://github.com/jfell13/RNA_Fold_Prediction.git
cd RNA_Fold_Prediction
pip install -r requirements.txt
```

Dependencies include:
- `torch`
- `torch_geometric`
- `numpy`
- `scikit-learn`
- `scipy`

## 🚀 Training

To train a model:
```bash
python train.py --model linear  # or 'gcn'
```

## 📊 Evaluation

After training, evaluate predictions using:
```bash
python evaluate.py --model gcn
```

Evaluation includes:
- RMSD calculation with optional Kabsch alignment
- Visual and numerical comparison to native coordinates

## 🧪 Example Results

Sample RMSD after 500 epochs:
- Linear Model: ~3.64 (training loss), RMSE ~2.82 Å
- GCN Model: In progress / under refinement

## 📌 Notes

- Dataset: Real RNA structures with C1′ coordinates from public sources or Kaggle
- The project is under active development.
- Additional models (e.g. SE(3)-Transformer, E(n)-GNN) may be included in future branches.

## 📈 Future Work

- Incorporate edge features (e.g., base pairing or dihedral angles)
- Explore SE(3)-equivariant architectures
- Improve data preprocessing and masking strategies for incomplete structures
- Upload model training checkpoints and visualizations

## 🧑‍💻 Author

**Jason Fell**  
Computational chemist exploring the intersection of structural biology and machine learning.  
[GitHub](https://github.com/jfell13)

---

## 📄 License

MIT License. See `LICENSE` for more information.
