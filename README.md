# ID3 Decision Tree Implementation

This repository contains an implementation of a decision tree from scratch using the ID3 algorithm. The model uses entropy to calculate information gain and determine the best splits. It is tested on the Car Evaluation dataset from the UCI Machine Learning Repository to classify cars based on categorical attributes.

Features
	•	Implementation of the ID3 algorithm from scratch.
	•	Entropy-based calculation of information gain.
	•	Preprocessing and handling of categorical data.
	•	Evaluation of the model’s performance on the Car Evaluation dataset.

Dataset

The Car Evaluation dataset is used to test the implementation. It evaluates cars based on attributes such as buying price, maintenance cost, number of doors, passenger capacity, and safety levels. The dataset is available at the UCI Machine Learning Repository.

How to Use

Clone the repository:
```
  git clone https://github.com/jvckdough/id3-decision-tree-car-evaluation.git
```

you can train and evaluate your model with:
```sh
python train.py -m decision_tree                   # runs with no depth limiting
python train.py -m decision_tree -d 2              # runs with depth_limit set to 2
```

You can run cross validation with:
```sh
python cross_validation.py -d 1 2 3 4              # runs CV with the depth_limit_values=[1, 2, 3, 4]

```

DATA: https://archive.ics.uci.edu/dataset/19/car+evaluation


