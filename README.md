# 📚 Dataset and Model Overview

This project evaluates a variety of machine learning models across diverse benchmark datasets, spanning domains such as biology, business, healthcare, and network security.

---

## 🧩 Dataset Summary

| **Dataset**       | **Domain**    | **Samples** | **Features** | **Description**                                          |
| ----------------- | ------------- | ----------- | ------------ | -------------------------------------------------------- |
| **iris**          | Biology       | 150         | 4            | Iris flower classification (3 species)                   |
| **diamonds**      | Business      | 53,940      | 9            | Diamond price prediction (regression task)               |
| **tips**          | Restaurant    | 244         | 6            | Tip amount prediction (regression task)                  |
| **titanic**       | History       | 891         | 11           | Titanic survival prediction (binary classification)      |
| **citeseer**      | Graph network | 3,312       | 3,703        | Paper citation network node classification (6 classes)   |
| **cora**          | Graph network | 2,708       | 1,433        | Paper citation network node classification (7 classes)   |
| **breast_cancer** | Medical       | 569         | 30           | Breast cancer diagnosis (benign vs. malignant)           |
| **digits**        | Image         | 1,797       | 64           | Handwritten digit recognition (10 classes: 0–9)          |
| **wine**          | Food & Drink  | 178         | 13           | Red wine quality classification (3 classes)              |
| **20newsgroups**  | NLP / Text    | 18,828      | 1,000        | News text classification (20 categories)                 |
| **kddcup99**      | Cybersecurity | 494,021     | 38           | Network intrusion detection (multi-class classification) |

---

## ⚙️ Supported Models

| **Model ID** | **Full Name**                           | **Type**                    | **Description**                                                         |
| ------------ | --------------------------------------- | --------------------------- | ----------------------------------------------------------------------- |
| `lr`         | Linear Regression / Logistic Regression | Regression / Classification | Simple linear model for regression or binary/multi-class classification |
| `svm`        | Support Vector Machine                  | Classification              | Kernel-based model for classification tasks                             |
| `rf`         | Random Forest                           | Classification              | Ensemble of decision trees for robust classification                    |
| `dt`         | Decision Tree                           | Classification              | Tree-based model for interpretable decisions                            |
| `gb`         | Gradient Boosting                       | Classification              | Boosted ensemble model for improved accuracy                            |
| `knn`        | K-Nearest Neighbors                     | Classification              | Distance-based non-parametric classifier                                |
| `mlp`        | Multi-Layer Perceptron                  | Classification              | Neural network for non-linear decision boundaries                       |

---

## 🧠 Model–Dataset Mapping

### 🔹 Regression Tasks  
Only regression models are applicable.

```python
"diamonds": ["lr"],   # Diamond price prediction | regression
"tips": ["lr"],       # Tip amount prediction | regression
