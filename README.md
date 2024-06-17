# Project 17 - COVID-19 Risk Prediction Project
## Project Overview
This project aims to predict the risk of severe outcomes for COVID-19 patients based on various features. The project involves data cleaning, exploratory data analysis (EDA), balancing the dataset, feature selection, model training, evaluation, and comparison using several machine learning algorithms.

## Project Structure
The project is organized as follows:
- clean_data(data): Cleans the input data by handling missing values, correcting errors, and creating a target variable.
- analyze_data(data): Performs exploratory data analysis to visualize the distribution of variables and their relationship with the target variable.
- balance_data(data, X, y): Balances the dataset to handle class imbalance.
- scalesplit_data(X, y): Splits the dataset into training and testing sets and scales the features.
- eval_performance(y_eval, X_eval, clf, clf_name): Evaluates the performance of a given classifier.
- plot_ROC_curve(models, plotname): Plots ROC curves for given models.

## Dependencies
The project requires the following Python libraries with their respective versions:
- imbalanced-learn (version 0.11.0)
- matplotlib (version 3.6.0)
- numpy (version 1.26.4)
- openpyxl (version 3.1.4)
- pandas (version 2.2.2)
- scikit-learn (version 1.4.2)
- scipy (version 1.13.1)
- seaborn (version 0.12.2)

You can install these dependencies using the provided `requirements.txt` file.

## Data Cleaning
The data cleaning process involves:
- Removing non-COVID patients.
- Correcting spelling errors.
- Handling missing values.
- Converting categorical variables to binary format.
- Creating a target variable AT_RISK.
- Dropping unnecessary columns.
- Handling gender-specific data issues.
- Dropping rows with remaining missing values.
- Converting data types to integers.
- Exploratory Data Analysis (EDA)

The EDA includes:
- Summary statistics of the dataset.
- Count plots of all variables.
- Distribution plot of the target variable.
- Heatmaps showing the relationship between categorical variables and the target variable.
- Boxplots of continuous variables with respect to the target variable.
- Correlation matrix to understand feature relationships.
- Analysis of class imbalance.
- Handling Class Imbalance
- The balance_data function uses RandomUnderSampler to balance the dataset and visualizes the distribution before and after resampling.

## Data Splitting and Scaling
The dataset is split into training and testing sets using a stratified split to maintain the class distribution. The features are then scaled using StandardScaler.

## Model Training and Evaluation
Several machine learning models are trained and evaluated, including:
- Random Forest Classifier (RF)
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Logistic Regression (LR)

Each model undergoes:
- Initial training and evaluation.
- Feature selection to retain the most important features.
- Hyperparameter tuning using GridSearchCV.
- Performance Evaluation

The performance of each model is evaluated using:
- Confusion matrices.
- ROC curves.
- Accuracy, precision, recall, F1-score, and ROC AUC score.

## Comparison of Models
The final performance of all models is compared using bar plots of the evaluation metrics and ROC curves.

## Output Files
The project generates several output files, including:
- Count plots (`output/countplots.png`)
- Target variable distribution (`output/count_target.png`)
- Heatmaps (`output/var_vs_target.png`)
- Boxplots (`utput/age_vs_target.png`)
- Correlation matrix (`output/corr_matrix.png`)
- Class distribution before and after balancing (`output/before_balance.png, output/after_balance.png`)
- Feature importances (`output/feature_importances_RF.png`)
- Confusion matrices for each model (`output/combined_confusion_matrices_RF.png`, `output/combined_confusion_matrices_SVM.png`, `output/combined_confusion_matrices_KNN.png`, `output/combined_confusion_matrices_LR.png`)
- Feature selection scores (`output/feature_selection_scores.png`)
- ROC curves (`output/roc_curves_RF_tuning.png`, `output/roc_curves_SVM_tuning.png`, `output/roc_curves_KNN_tuning.png`, `output/roc_curves_LR_tuning.png`, `output/ROC_comparison.png`)
- Performance summary Excel file (`output/performance_summary.xlsx`)
- Performance comparison plots (`output/Confusion_Matrix_Metrics_Comparison.png`, `output/Performance_Metrics_Comparison.png`)

## Running the Project
- Ensure all dependencies are installed.
- Place the dataset file (dataset.csv) in the project directory.
- Verify that an output folder exists in the project directory to save the output files. If it doesn't exist, create it.
- Run the Python script `MLA_Project17_G4-B.py`to execute the entire workflow.
