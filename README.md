# Network Intrusion Detection – Machine Learning Challenge

## About the Project

This project focuses on detecting and classifying network traffic using machine learning techniques.

It was developed as part of the Kaggle Network Intrusion Detection ML Challenge.

The model classifies network traffic into four categories:

- **Benign** – Normal network activity.
- **Availability** – Availability-related attacks.
- **Reconnaissance** – Network scanning and information gathering.
- **Spoofing** – Spoofing-related network activity.

## Competition Details

- **Platform:** Kaggle
- **Task:** Multiclass Classification
- **Evaluation Metric:** Macro F1 Score
- **Training Dataset:** 84,713 records
- **Testing Dataset:** 33,161 records
- **Input Features:** 82 original features

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- LightGBM
- XGBoost
- CatBoost

## Methodology

The project involves the following stages:

1. Data loading and exploration.
2. Data preprocessing and missing-value handling.
3. Network traffic feature engineering.
4. Categorical feature encoding.
5. Machine learning model training.
6. Ensemble learning and prediction.
7. Submission file generation.

## Project Structure

```text
network-intrusion-detection-ml/
│
├── network_intrusion.py
├── network_intrusion.ipynb
├── requirements.txt
└── README.md
```

## Installation

Install the required Python libraries:

```bash
pip install -r requirements.txt
```

## Execution

Place the Kaggle competition datasets in the appropriate input directory and execute:

```bash
python network_intrusion.py
```

The notebook can also be executed using Kaggle Notebooks.

## Output

The model generates a CSV prediction file with the following columns:

- `id`
- `Target`

## Evaluation

The competition uses the **Macro F1 Score** to evaluate multiclass classification performance.

## Author

**Radha Krishna**

B.Tech – Computer Science and Engineering (IoT)

VNR VJIET, Hyderabad
