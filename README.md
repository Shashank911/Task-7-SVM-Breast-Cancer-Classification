Breast Cancer Classification using Support Vector Machines (SVM)

Overview
--------
This project implements Support Vector Machine (SVM) models to classify breast cancer tumors as benign or malignant using the Breast Cancer Dataset from Kaggle.
The goal is to:
- Compare Linear and RBF kernel SVM models.
- Optimize hyperparameters for improved performance.
- Evaluate and rank models based on multiple metrics.
- Generate visualizations for interpretability.

About the Dataset
-----------------
Source: Kaggle Breast Cancer Dataset  [https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset]
Description:
- Contains measurements from breast mass cell nuclei obtained via digitized images.
- Features: 30 numerical attributes describing tumor cell properties (e.g., radius, texture, smoothness).
- Target variable:
  - 0 → Malignant (cancerous)
  - 1 → Benign (non-cancerous)
- No missing values; ready for preprocessing.

Workflow
--------
1. **Data Loading & Exploration**
   - Load dataset into Pandas DataFrame.
   - Inspect data shape, missing values, and class distribution.

2. **Data Preprocessing**
   - Encode categorical labels (Malignant/Benign → 0/1).
   - Feature scaling using StandardScaler.

3. **Model Building**
   - Train baseline SVM models (Linear and RBF kernels).
   - Perform hyperparameter tuning using GridSearchCV.

4. **Model Evaluation**
   - Calculate Accuracy, Precision, Recall, F1-score, and ROC-AUC.
   - Plot Confusion Matrix and ROC Curves.
   - Save evaluation plots in `outputs/` directory.

5. **Model Selection**
   - Rank models primarily by F1-score and ROC-AUC.
   - Select best model for predictions.

6. **Results & Insights**
   - Compare tuned vs baseline performance.
   - Discuss findings and practical applications.

Tools & Libraries Used
----------------------
- **Python 3.x**
- **Jupyter Notebook** — interactive coding environment.
- **pandas** — data manipulation and analysis.
- **numpy** — numerical computations.
- **scikit-learn** — SVM models, preprocessing, metrics, and hyperparameter tuning.
- **matplotlib** — data visualization.
- **seaborn** — advanced plotting and styling.

Project Structure
-----------------
breast-cancer-svm/
│
├── SVM_BreastCancer.ipynb   # Main Jupyter Notebook
├── outputs/                 # Saved plots and results
├── breast_cancer.csv        # Dataset (download from Kaggle)
└── README.txt               # Project documentation

Usage
-----
1. Place `breast_cancer.csv` in the project directory.
2. Open `SVM_BreastCancer.ipynb` in Jupyter Notebook.
3. Run cells sequentially to:
   - Load and preprocess data.
   - Train and evaluate SVM models.
   - View plots and metrics in the `outputs/` folder.

Results
-------
Example performance (may vary depending on train/test split):

Model              | Accuracy | Precision | Recall | F1-score | ROC-AUC
-------------------|----------|-----------|--------|----------|--------
Linear SVM         | 96.4%    | 95%       | 97%    | 96%      | 0.98
RBF SVM (Tuned)    | 97.1%    | 96%       | 98%    | 97%      | 0.99

Conclusion
----------
SVM models, when properly tuned, deliver highly accurate and robust results for medical classification tasks.
This work highlights the potential of machine learning to assist healthcare professionals in early and reliable diagnosis.

Acknowledgment - This project is part of my AI/ML Internship to gain hands-on experience with classification algorithms using real-world datasets.


Author
------
Shashank Chauhan
Email: cshashank899@gmail.com.com
GitHub: https://github.com/Shashank911
