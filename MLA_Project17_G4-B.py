# General
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score, KFold
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif, RFE
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix, roc_auc_score, precision_recall_curve,  r2_score, mean_squared_error, classification_report
from sklearn.ensemble import RandomForestClassifier
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn import svm, model_selection
from sklearn.pipeline import Pipeline

# scipystats
import scipy.stats as sts
from scipy.stats import t, ttest_ind


###################################### FUNCTIONS ################################################

# Data Cleaning
def clean_data(data):
    print("### CLEANING DATA ###")

    data_copy = data.copy()

    #1) Drop non-covid patients -> rows where "CLASIFFICATION_FINAL" is greater than or equal to 4
    data_copy = data_copy[data_copy["CLASIFFICATION_FINAL"] < 4]
    data_copy = data_copy.drop(columns=['CLASIFFICATION_FINAL'])

    # 2) CORRECT SPELLING ERROR
    data_copy.rename(columns={'HIPERTENSION': 'HYPERTENSION'}, inplace=True)

    # 3) CHANGE 97, 98 and 99 to NaN (=missing values)
    boolean_features = data_copy.columns.drop(["AGE", "MEDICAL_UNIT", "DATE_DIED"])
    data_copy[boolean_features] = data_copy[boolean_features].replace({97: np.nan, 98: np.nan, 99: np.nan})

    #Overview before cleaning
    print()
    print("Missing values before cleaning data: ")
    print(data_copy.isna().sum())

    # 4) CHANGE TO BINARY: REPLACE '2' WITH '0' (boolean features), SO 0 = NO, 1 = YES (BC "In the Boolean features, 1 means "yes" and 2 means "no")
    data_copy[boolean_features] = data_copy[boolean_features].replace({2: 0})
    # BINARY SEX: REPLACE 2 WITH 0, SO 0 = MALE, 1 = FEMALE

    # 5) FEATURE DATE_DIED INTO BINARY INSTEAD, if condition #TRUE = 1, #FALSE = 0
    data_copy['DEAD'] = (data_copy['DATE_DIED'] != '9999-99-99').astype(int)
    data_copy = data_copy.drop(columns=['DATE_DIED'])

    # 6) INTRODUCE OUR TARGET VARIABLE Y
    data_copy['AT_RISK'] = data_copy['DEAD'] + data_copy['ICU'].fillna(0) + data_copy['INTUBED'].fillna(0)  # DEAD HAS NO MISSING VALUES, SO IT SHOULD ALWAYS BE AT LEAST 0. WE CAN THEREFORE SIMPLY CHANGE THE NaN TO 0 TO ADD.
    data_copy['AT_RISK'] = data_copy['AT_RISK'].apply(lambda x: 1 if x > 0 else 0)
    data_copy = data_copy.drop(columns=['INTUBED', 'ICU', 'DEAD'])  # bc now as AT_RISK

    # 7) PREGNANT: MALE PATIENTS MUST HAVE MANY NAN: REPLACE WITH 0 IF PATIENT WITH NAN IS A MALE
    # Replace missing values in 'PREGNANT' column with 2 where 'SEX' column is equal to 2 or (SEX = 1 and AGE >= 60)
    data_copy.loc[(data_copy['SEX'] == 0) | ((data_copy['SEX'] == 1) & (data_copy['AGE'] >= 60)), 'PREGNANT'] = 0

    # Check filtering pregnancy
    print()
    print("Filter pregnant: ")
    print(data_copy.isna().sum())  # check again

    # 8) DROP COLS: THEY DON'T HAVE EFFECT IF PATIENT IS AT RISK DUE TO RISK FACTOR, THEY ARE THE RESULT OF IT
    data_copy = data_copy.drop(columns=['MEDICAL_UNIT', 'PATIENT_TYPE', 'USMER'])

    # 9) DROP ROWS OF MISSING VALUES:
    data_copy.dropna(axis=0, inplace=True)

    # 10) CONVERT TO INTEGER
    data_copy = data_copy.astype(int)

    #Overview after cleaning
    print()
    print("Missing values after cleaning: ")
    print(data_copy.isna().sum())
    print(data_copy.dtypes)
    print(data_copy.shape)

    print()
    print("### CLEANING DONE ###")

    return data_copy


# Exploratory Data Analysis (EDA)
def analyze_data(data):
    print()
    print("### EXPLORATORY DATA ANALYSIS ###")

    # 1) Get summary statistics
    print()
    print("Summary statistics: ")
    print(data.describe(include='all'))

    # 2) Visualize countplots of all variables
    fig, axes = plt.subplots(3, 5, figsize=(15, 9))
    axes = axes.flatten()
    fig.suptitle('Countplots of Variables')
    for i, col in enumerate(data.columns):
        sns.countplot(ax=axes[i], x=data[col])
        axes[i].set_title(f'Countplot of {col}')
    plt.tight_layout()
    plt.savefig("output/countplots.png")
    plt.close(fig)

    # 3) Visualize target variable
    plt.figure(figsize=(6, 4))
    ax = sns.countplot(x='AT_RISK', data=data)
    plt.bar_label(ax.containers[0])
    plt.title('Distribution of Risk')
    plt.xlabel('AT_RISK')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig("output/count_target.png")
    plt.close()

    # 4) Visualize categorical Variables
    fig, axes = plt.subplots(3, 5,figsize=(18, 12), gridspec_kw={'hspace': 0.5, 'wspace': 0.5})
    axes = axes.flatten()
    fig.suptitle('Heatmaps of Variables with respect to AT_RISK')
    categorical_columns = data.columns.drop(["AGE", "AT_RISK"])
    for i, col in enumerate(categorical_columns):
        cross_tab = pd.crosstab(data[col], data['AT_RISK'], dropna=False)
        #plt.figure(figsize=(8, 6))
        sns.heatmap(cross_tab, annot=True, fmt='d', cmap='YlGnBu', ax=axes[i])
        axes[i].set_title(f'Heatmap of {col}')
    # Remove any empty subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    fig.subplots_adjust(top=0.95)
    plt.savefig("output/var_vs_target.png")
    plt.close(fig)

    # 5) Visualize Continuous Variables
    continuous_columns = ["AGE"]
    for col in continuous_columns:
        plt.figure(figsize=(8, 6))
        sns.boxplot(x=data['AT_RISK'], y=col, data=data)
        plt.title(f'Distribution of {col} with respect to Risk')
        plt.xlabel('AT_RISK')
        plt.ylabel(col)
    plt.savefig("output/age_vs_target.png")
    plt.close()

    # 6) Correlation Analysis
    correlation_matrix = data.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Correlation Matrix')
    plt.savefig("output/corr_matrix.png")
    plt.close()

    # 7) Class imbalance
    # Check positive class prevalence
    # mean of the 'AT_RISK' column effectively gives the proportion of positive instances in the dataset
    data['AT_RISK'].mean()
    # total positive instances
    data['AT_RISK'].sum()
    print('Data set comprises {:.0f} positive instances, implying a positive class prevalence of {:.3f}'.format(data['AT_RISK'].sum(),data['AT_RISK'].mean()))

    return


# Remove class imbalance
def balance_data(data, X, y):
    print()
    print("### BALANCE DATA ###")

    #plot before
    plt.figure(figsize=(8, 6))
    ax = sns.countplot(x=data['AT_RISK'], palette='ch:start=.2,rot=-.3')
    plt.bar_label(ax.containers[0])
    plt.title('Risk Distribution', fontsize=18)
    plt.savefig("output/before_balance.png")

    #undersample
    rand_under = RandomUnderSampler(random_state=1)
    x_resampled, y_resampled = rand_under.fit_resample(X, y)

    #plot after
    plt.figure(figsize=(8, 6))
    ax = sns.countplot(x=y_resampled, palette='ch:start=.2,rot=-.3')
    plt.bar_label(ax.containers[0])
    plt.title("Risk Distribution After Resampling", fontsize=16)
    plt.savefig("output/after_balance.png")

    #Overview
    print()
    print("Data after resampling: ")
    print("X: ", x_resampled.shape)
    print("y: ", y_resampled.shape)

    return x_resampled, y_resampled


# Split and scale data
def scalesplit_data(X, y):
    print()
    print("### SPLITTING & SCALING ###")

    # Split the dataset into training and testing sets, 80% for training and 20% for testing
    # Use a stratified split to account for class imbalance in outcome! -> same proportion of pos. class in both training & testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

    # Feature scaling
    scaler = StandardScaler()
    X_train[['AGE']] = scaler.fit_transform(X_train[['AGE']])
    X_test[['AGE']] = scaler.transform(X_test[['AGE']])

    print()
    print("X_train :", X_train.shape)
    print("X_test :", X_test.shape)
    print("y_train :", y_train.shape)
    print("y_test :", y_test.shape)

    return X_train, X_test, y_train, y_test


# Evaluate ML performance
def eval_performance(y_eval, X_eval, clf, clf_name = 'My classifier'):
    y_pred = clf.predict(X_eval)
    y_pred_proba = clf.predict_proba(X_eval)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_eval, y_pred).ravel()

    # Evaluation
    accuracy = accuracy_score(y_eval, y_pred)
    precision = precision_score(y_eval, y_pred)
    recall = recall_score(y_eval, y_pred)
    f1 = f1_score(y_eval, y_pred)
    fp_rates, tp_rates, _ = roc_curve(y_eval, y_pred_proba)

    # Calculate the area under the roc curve using a sklearn function
    roc_auc = auc(fp_rates, tp_rates)

    conf_matrix = confusion_matrix(y_eval, y_pred)

    return y_pred, fp_rates, tp_rates, conf_matrix, (tp, fp, tn, fn, accuracy, precision, recall, f1, roc_auc)


#Plot ROC curve
def plot_ROC_curve(models, plotname):
    plt.figure(figsize=(10, 8))
    for name, (fpr, tpr, roc_auc) in models.items():
        # Plot ROC curve
        plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.2f})')

    # Plot the random guess line
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Random Guess', alpha=.8)
    # Configure plot
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    plt.grid()
    # Save the plot
    plt.tight_layout()
    plt.savefig(f'output/roc_curves_{plotname}.png')
    plt.show()


###################################### START OF CODE ################################################

## Data exploration, cleaning, analysis
data = pd.read_csv('dataset.csv', header=0)

data_first_20_rows = data.iloc[:20]
data_first_20_rows.to_csv('first_20_rows.csv')

#Data exploration
print()
print("### DATA EXPLORATION ### ")
print("Shape: ", data.shape)
print("Columns: ", data.columns)
print(data.head(5))
print()

#Cleaning
data = clean_data(data)

data_cleaned_first_20_rows = data.iloc[:20]
data_cleaned_first_20_rows.to_csv('clean_first_20_rows.csv')

#Data analysis
analyze_data(data)

## Define features & label, balance data, split & scale, fit 1st model
X = data.drop(['AT_RISK'], axis=1)
y = data['AT_RISK']

#Balance data
X_balanced, y_balanced = balance_data(data, X, y)

#Split and scale data using selected features
X_train, X_test, y_train, y_test = scalesplit_data(X_balanced, y_balanced)


###################################### MODELS ################################################


###################################### RF ################################################

## Random Forest 1st fit
print()
print("### RF 1ST FIT ###")
clf_RF = RandomForestClassifier()
clf_RF.fit(X_train, y_train)

# Evaluate performance 1
df_performance_RF = pd.DataFrame(columns = ['tp','fp','tn','fn','accuracy', 'precision', 'recall', 'f1', 'roc_auc'] )
y_pred_test, fp_rates_test, tp_rates_test, conf_matrix_rf_test, df_performance_RF.loc['RF (test) - default',:] = eval_performance(y_test, X_test, clf_RF, clf_name = 'RF (test)')
y_pred_train, fp_rates_train, tp_rates_train, conf_matrix_rf_train, df_performance_RF.loc['RF (train) - default',:] = eval_performance(y_train, X_train, clf_RF, clf_name = 'RF (train)')
print()
print("### PERFORMANCE EVALUATION ###")
print(df_performance_RF)


## Feature selection RF
#Get feature importances
feature_importances = clf_RF.feature_importances_
# Create a DataFrame with feature importances and corresponding column names
feature_importance_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': feature_importances
})
# Sort the DataFrame by importance in descending order
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
print()
print("### FEATURE SELECTION ###")
print("Feature importances:\n", feature_importance_df)

#Plot
plt.figure(figsize=(12, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title("Feature Importance in Random Forest")
plt.tight_layout()
plt.savefig('output/feature_importances_RF.png')


#Drop least important features
threshold=0.015
selected_features = feature_importance_df[feature_importance_df['Importance'] >= threshold]['Feature']
print()
print("Selected features: ", selected_features)

# Filter X_train and X_test to include only the selected features
X_train_sel_RF = pd.DataFrame(X_train, columns=X.columns)[selected_features]
X_test_sel_RF = pd.DataFrame(X_test, columns=X.columns)[selected_features]


## Random Forest 2nd fit
print()
print("### RF 2ND FIT (SELECTED FEATURES) ###")
clf_RF_selected = RandomForestClassifier()
clf_RF_selected.fit(X_train_sel_RF, y_train)

# Evaluate performance 2
y_pred_sel_test, fp_rates_sel_test, tp_rates_sel_test, conf_matrix_rf_sel_test, df_performance_RF.loc['RF (test) - selected',:] = eval_performance(y_test, X_test_sel_RF, clf_RF_selected, clf_name = 'RF selected (test)')
y_pred_sel_train, fp_rates_sel_train, tp_rates_sel_train, conf_matrix_rf_sel_train, df_performance_RF.loc['RF (train) - selected',:] = eval_performance(y_train, X_train_sel_RF, clf_RF_selected, clf_name = 'RF selected (train)')
print()
print("### PERFORMANCE EVALUATION ###")
print(df_performance_RF)


## Hyperparameter tuning
# Random forest hyperparameters:
# Number of trees in the forest (n_estimators)
# Function to measure quality of a split (criterion)
# Number of features to consider when looking for the best split (max_features)
# Maximum depth of the trees (max_depth)
# Minimum number of samples required to split an internal node (min_samples_split)
# Minimum number of samples required to be at a leaf node (min_samples_leaf)
param_grid = {
    'n_estimators': [200, 500],         #tried 50, 100, 200, 500
    'criterion': ['gini', 'entropy'],   #tried gini, entropy, log_loss
    'max_features': ['sqrt', 'log2'],   #tried sqrt, None, auto, log2
    'max_depth': [2, 5],                # tried 2, 5, 10
    'min_samples_split': [2, 5]         # tried 2, 5
}
# Set up the grid search
# Recall/Sensitivity most important -> better to plan more resources than not enough
grid_search = GridSearchCV(estimator=clf_RF, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2, scoring='recall')

# Fit the grid search to the data
grid_search.fit(X_train_sel_RF, y_train)

# Print the best parameters and best score
print("Best parameters found: ", grid_search.best_params_)
print("Best recall score found: ", grid_search.best_score_)

# Get the best model
best_clf_RF = grid_search.best_estimator_

# Evaluate performance 3
y_pred_sel_tune_test, fp_rates_sel_tune_test, tp_rates_sel_tune_test, conf_matrix_rf_sel_tune_test, df_performance_RF.loc['RF (test) - tuned',:] = eval_performance(y_test, X_test_sel_RF, best_clf_RF, clf_name = 'RF selected & tuned (test)')
y_pred_sel_tune_train, fp_rates_sel_tune_train, tp_rates_sel_tune_train, conf_matrix_rf_sel_tune_train, df_performance_RF.loc['RF (train) - tuned',:] = eval_performance(y_train, X_train_sel_RF, best_clf_RF, clf_name = 'RF selected & tuned (train)')
print()
print("### PERFORMANCE EVALUATION ###")
print(df_performance_RF)


# Plotting confusion matrices
conf_matrices = [
    (conf_matrix_rf_test, 'RF (test)'),
    (conf_matrix_rf_train, 'RF (train)'),
    (conf_matrix_rf_sel_test, 'RF selected (test)'),
    (conf_matrix_rf_sel_train, 'RF selected (train)'),
    (conf_matrix_rf_sel_tune_test, 'RF selected & tuned (test)'),
    (conf_matrix_rf_sel_tune_train, 'RF selected & tuned (train)')
]

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Confusion Matrices for Different Models')

for ax, (conf_matrix, title) in zip(axes.flat, conf_matrices):
    sns.heatmap(conf_matrix, annot=True, fmt='d', ax=ax, cmap='Blues')
    ax.set_title(title)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('output/combined_confusion_matrices_RF.png')


# Plotting ROC curves
roc_data = {
    'RF (test)': (fp_rates_test, tp_rates_test, df_performance_RF.loc['RF (test) - default', 'roc_auc']),
    'RF (train)': (fp_rates_train, tp_rates_train, df_performance_RF.loc['RF (train) - default', 'roc_auc']),
    'RF selected (test)': (fp_rates_sel_test, tp_rates_sel_test, df_performance_RF.loc['RF (test) - selected', 'roc_auc']),
    'RF selected (train)': (fp_rates_sel_train, tp_rates_sel_train, df_performance_RF.loc['RF (train) - selected', 'roc_auc']),
    'RF selected & tuned (test)': (fp_rates_sel_tune_test, tp_rates_sel_tune_test, df_performance_RF.loc['RF (test) - tuned', 'roc_auc']),
    'RF selected & tuned (train)': (fp_rates_sel_tune_train, tp_rates_sel_tune_train, df_performance_RF.loc['RF (train) - tuned', 'roc_auc'])
}

# Plot ROC curves
plot_ROC_curve(roc_data, 'RF_tuning')

###################################### SVM ################################################

#Sample to speed it up
sample_size = 10000
X_train_sample = X_train[:sample_size]
y_train_sample = y_train[:sample_size]

# Create and train the SVM model
svm_model = svm.SVC(kernel='linear', probability=True, max_iter= 10000)  # Using linear kernel
svm_model.fit(X_train_sample, y_train_sample)

# Evaluate performance
df_performance_SVM = pd.DataFrame(columns = ['tp','fp','tn','fn','accuracy', 'precision', 'recall', 'f1', 'roc_auc'] )
y_pred_SVM_test, fpr_SVM_test, tpr_SVM_test, conf_matrix_SVM_test, df_performance_SVM.loc['SVM (test) - default',:] = eval_performance(y_test, X_test, svm_model, clf_name = 'SVM (test)')
y_pred_SVM_train, fpr_SVM_train, tpr_SVM_train, conf_matrix_SVM_train, df_performance_SVM.loc['SVM (train) - default',:] = eval_performance(y_train_sample, X_train_sample, svm_model, clf_name = 'SVM (train)')
print()
print("### PERFORMANCE EVALUATION ###")
print(df_performance_SVM)

# Hyperparameter tuning with GridSearchCV
param_grid = {'C': [0.1, 10], 'kernel': ['linear', 'rbf']}
grid_search = GridSearchCV(svm.SVC(probability=True, max_iter=10000), param_grid, refit=True, cv=5, n_jobs=-1, verbose=2, scoring='recall')
grid_search.fit(X_train_sample, y_train_sample)

# Best parameters and model evaluation
print(f"Best parameters found: {grid_search.best_params_}")
best_svm_model = grid_search.best_estimator_
y_pred_SVM_tune_test, fpr_SVM_tune_test, tpr_SVM_tune_test, conf_matrix_SVM_tune_test, df_performance_SVM.loc['SVM (test) - tuned',:] = eval_performance(y_test, X_test, best_svm_model, clf_name = 'SVM tuned (test)')
y_pred_SVM_tune_train, fpr_SVM_tune_train, tpr_SVM_tune_train, conf_matrix_SVM_tune_train, df_performance_SVM.loc['SVM (train) - tuned',:] = eval_performance(y_train_sample, X_train_sample, best_svm_model, clf_name = 'SVM tuned (train)')
print()
print("### PERFORMANCE EVALUATION ###")
print(df_performance_SVM)

# Plotting confusion matrices
conf_matrices = [
    (conf_matrix_SVM_test, 'SVM (test) - default'),
    (conf_matrix_SVM_train, 'SVM (train) - default'),
    (conf_matrix_SVM_tune_test, 'SVM (test) - tuned'),
    (conf_matrix_SVM_tune_train, 'SVM (train) - tuned')
]

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Confusion Matrices for Different Models')

for ax, (conf_matrix, title) in zip(axes.flat, conf_matrices):
    sns.heatmap(conf_matrix, annot=True, fmt='d', ax=ax, cmap='Blues')
    ax.set_title(title)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('output/combined_confusion_matrices_SVM.png')

# Plotting ROC curves
roc_data = {
    'SVM (test) - default': (fpr_SVM_test, tpr_SVM_test, df_performance_SVM.loc['SVM (test) - default', 'roc_auc']),
    'SVM (train) - default': (fpr_SVM_train, tpr_SVM_train, df_performance_SVM.loc['SVM (train) - default', 'roc_auc']),
    'SVM (test) - tuned': (fpr_SVM_tune_test, tpr_SVM_tune_test, df_performance_SVM.loc['SVM (test) - tuned', 'roc_auc']),
    'SVM (train) - tuned': (fpr_SVM_tune_train, tpr_SVM_tune_train, df_performance_SVM.loc['SVM (train) - tuned', 'roc_auc'])
}

# Plot ROC curves
plot_ROC_curve(roc_data, 'SVM_tuning')


###################################### KNN ################################################

select_k_best = SelectKBest(score_func=f_classif, k=4)

# K-NEAREST NEIGHBORS MODEL: proximity to make classifications/predictions
knn_model = KNeighborsClassifier()

# Pipeline erstellen
pipeline = Pipeline([
    ('feature_selection', select_k_best),
    ('knn', knn_model)
])

# Default
knn_model.fit(X_train, y_train)
df_performance_KNN = pd.DataFrame(columns = ['tp','fp','tn','fn','accuracy', 'precision', 'recall', 'f1', 'roc_auc'] )
y_pred_KNN_test, fpr_KNN_test, tpr_KNN_test, conf_matrix_KNN_test, df_performance_KNN.loc['KNN (test) - default',:] = eval_performance(y_test, X_test, knn_model, clf_name = 'KNN (test)')
y_pred_KNN_train, fpr_KNN_train, tpr_KNN_train, conf_matrix_KNN_train, df_performance_KNN.loc['KNN (train) - default',:] = eval_performance(y_train, X_train, knn_model, clf_name = 'KNN (train)')
print()

# GridSearchCV to decide on number of neighbors #due to computational efficiency between 1 and 20, took forever and never completed with up to 50
parameter_grid = {'knn__n_neighbors': [400, 600]} #first tried with 100, 500, 1000, best k = 500
grid_search = GridSearchCV(pipeline, parameter_grid, cv=5, n_jobs=-1, verbose=2, scoring='recall')
grid_search.fit(X_train, y_train)

# BEST KNN MODEL found with gridsearch
best_knn_model = grid_search.best_estimator_
print(f"Best k: {grid_search.best_params_['knn__n_neighbors']}") #400

# Evaluate performance
#df_performance_KNN = pd.DataFrame(columns = ['tp','fp','tn','fn','accuracy', 'precision', 'recall', 'f1', 'roc_auc'] )
y_pred_KNN_tune_test, fpr_KNN_tune_test, tpr_KNN_tune_test, conf_matrix_KNN_tune_test, df_performance_KNN.loc['KNN (test) - selected & tuned',:] = eval_performance(y_test, X_test, best_knn_model, clf_name = 'KNN tuned (test)')
y_pred_KNN_tune_train, fpr_KNN_tune_train, tpr_KNN_tune_train, conf_matrix_KNN_tune_train, df_performance_KNN.loc['KNN (train) - selected & tuned',:] = eval_performance(y_train, X_train, best_knn_model, clf_name = 'KNN tuned (train)')
print()
print("### PERFORMANCE EVALUATION ###")
print(df_performance_KNN)
print(df_performance_KNN['accuracy'])

# Plotting confusion matrices
conf_matrices = [
    (conf_matrix_KNN_test, 'KNN (test) - default'),
    (conf_matrix_KNN_train, 'KNN (train) - default'),
    (conf_matrix_KNN_tune_test, 'KNN (test) - selected & tuned'),
    (conf_matrix_KNN_tune_train, 'KNN (train) - selected & tuned')
]

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Confusion Matrices for Different Models')

for ax, (conf_matrix, title) in zip(axes.flat, conf_matrices):
    sns.heatmap(conf_matrix, annot=True, fmt='d', ax=ax, cmap='Blues')
    ax.set_title(title)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('output/combined_confusion_matrices_KNN.png')

feature_names = data.columns[:-1]

# Plotting feature selection scores
X_train_selected = select_k_best.fit_transform(X_train, y_train)
feature_scores = select_k_best.scores_
selected_features_indices = select_k_best.get_support(indices=True)
selected_features_scores = feature_scores[selected_features_indices]

# Assuming feature names are available as a list 'feature_names'
selected_feature_names = [feature_names[i] for i in selected_features_indices]

plt.figure(figsize=(12, 6))
sns.barplot(x=selected_feature_names, y=selected_features_scores)
plt.xticks(rotation=90)
plt.xlabel('Features')
plt.ylabel('Scores')
plt.title('Feature Selection Scores (SelectKBest)')
plt.tight_layout()
plt.savefig('output/feature_selection_scores.png')
plt.show()

# Plotting ROC curves
roc_data = {
    'KNN (test) - default': (fpr_KNN_test, tpr_KNN_test, df_performance_KNN.loc['KNN (test) - default', 'roc_auc']),
    'KNN (train) - default': (fpr_KNN_train, tpr_KNN_train, df_performance_KNN.loc['KNN (train) - default', 'roc_auc']),
    'KNN (test) - selected & tuned': (fpr_KNN_tune_test, tpr_KNN_tune_test, df_performance_KNN.loc['KNN (test) - selected & tuned', 'roc_auc']),
    'KNN (train) - selected & tuned': (fpr_KNN_tune_train, tpr_KNN_tune_train, df_performance_KNN.loc['KNN (train) - selected & tuned', 'roc_auc'])
}

# Plot ROC curves
plot_ROC_curve(roc_data, 'KNN_tuning')


###################################### LR ################################################

# Model fitting
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

# first Evaluation (before feature selection and tuning)
df_performance_LR = pd.DataFrame(columns = ['tp','fp','tn','fn','accuracy', 'precision', 'recall', 'f1', 'roc_auc'] )
y_pred_LR_test, fpr_LR_test, tpr_LR_test, conf_matrix_LR_test, df_performance_LR.loc['LR (test) - default',:] = eval_performance(y_test, X_test, logreg, clf_name = 'LR (test)')
y_pred_LR_train, fpr_LR_train, tpr_LR_train, conf_matrix_LR_train, df_performance_LR.loc['LR (train) - default',:] = eval_performance(y_train, X_train, logreg, clf_name = 'LR (train)')
print()
print("### PERFORMANCE EVALUATION: Default ###")
print(df_performance_LR)

# Feature Selection using RFE with cross-validation
scorer = make_scorer(accuracy_score)
n_features_range = range(1, X_train.shape[1] + 1)

best_score = 0
best_n_features = 0

for n_features in n_features_range:
    rfe_LR = RFE(logreg, n_features_to_select=n_features)
    scores = cross_val_score(rfe_LR, X_train, y_train, cv=5, scoring=scorer)
    mean_score = np.mean(scores)
    print(f"Number of features: {n_features}, Cross-validation score: {mean_score:.4f}")

    if mean_score > best_score:
        best_score = mean_score
        best_n_features = n_features

print(f"Best number of features: {best_n_features}, Best cross-validation score: {best_score:.4f}")

# RFE with best number of features
rfe_LR = RFE(logreg, n_features_to_select=best_n_features)

# transform sets
X_train_sel_LR = rfe_LR.fit_transform(X_train, y_train)
X_test_sel_LR = rfe_LR.transform(X_test)

# Model fitting with best features
logreg_sel = LogisticRegression()
logreg_sel.fit(X_train_sel_LR, y_train)

# Evaluate performance with best features
y_pred_sel_LR_test, fpr_sel_LR_test, tpr_sel_LR_test, conf_matrix_LR_sel_test, df_performance_LR.loc['LR (test) - selected', :] = eval_performance(y_test, X_test_sel_LR, logreg_sel, clf_name='LR (test) - selected')
y_pred_sel_LR_train, fpr_sel_LR_train, tpr_sel_LR_train, conf_matrix_LR_sel_train, df_performance_LR.loc['LR (train) - selected', :] = eval_performance(y_train, X_train_sel_LR, logreg_sel, clf_name='LR (train) - selected')

print()
print("### PERFORMANCE EVALUATION: RFE ###")
print(df_performance_LR)

# Hyperparameter tuning with GridSearchCV
param_grid_LR = {
    'penalty': ['l1', 'l2 '],
    'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['liblinear'],
    'max_iter': [100, 200, 300]
}

grid_search_LR = GridSearchCV(estimator=logreg, param_grid=param_grid_LR, cv=5, n_jobs=-1, verbose=2, scoring='recall')
grid_search_LR.fit(X_train_sel_LR, y_train)

# Best model
print(f"Best parameters found: {grid_search_LR.best_params_}")
best_LR_model = grid_search_LR.best_estimator_

# Evaluation with tuning
y_pred_LR_tune_test, fpr_LR_tune_test, tpr_LR_tune_test, conf_matrix_LR_tune_test, df_performance_LR.loc['LR (test) - tuned', :] = eval_performance(y_test, X_test_sel_LR, best_LR_model, clf_name='LR selected & tuned (test)')
y_pred_LR_tune_train, fpr_LR_tune_train, tpr_LR_tune_train, conf_matrix_LR_tune_train, df_performance_LR.loc['LR (train) - tuned', :] = eval_performance(y_train, X_train_sel_LR, best_LR_model, clf_name='LR selected & tuned (train)')

print("### PERFORMANCE EVALUATION: Selected & Tuned###")
print(df_performance_LR)

# Plotting confusion matrices
conf_matrices_LR = [
    (conf_matrix_LR_test, 'LR (test) - default'),
    (conf_matrix_LR_train, 'LR (train) - default'),
    (conf_matrix_LR_sel_test, 'LR (test) - selcted'),
    (conf_matrix_LR_sel_train, 'LR (train) - selected'),
    (conf_matrix_LR_tune_test, 'LR (test) - tuned'),
    (conf_matrix_LR_tune_train, 'LR (train) - tuned'),
]

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Confusion Matrices for Different Models')

for ax, (conf_matrix, title) in zip(axes.flat, conf_matrices_LR):
    sns.heatmap(conf_matrix, annot=True, fmt='d', ax=ax, cmap='Blues')
    ax.set_title(title)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('output/combined_confusion_matrices_LR.png')

# Plotting ROC curves
roc_data = {
    'LR (test)': (fpr_LR_test, tpr_LR_test, df_performance_LR.loc['LR (test) - default', 'roc_auc']),
    'LR (train)': (fpr_LR_train, tpr_LR_train, df_performance_LR.loc['LR (train) - default', 'roc_auc']),
    'LR selected (test)': (fpr_sel_LR_test, tpr_sel_LR_test, df_performance_LR.loc['LR (test) - selected', 'roc_auc']),
    'LR selected (train)': (fpr_sel_LR_train, tpr_sel_LR_train, df_performance_LR.loc['LR (train) - selected', 'roc_auc']),
    'LR selected & tuned (test)': (fpr_LR_tune_test, tpr_LR_tune_test, df_performance_LR.loc['LR (test) - tuned', 'roc_auc']),
    'LR selected & tuned (train)': (fpr_LR_tune_train, tpr_LR_tune_train, df_performance_LR.loc['LR (train) - tuned', 'roc_auc'])
}

plot_ROC_curve(roc_data, 'LR_tuning')

###################################### Compare Model Performance ################################################
# Collect all df_performance dataframes in a dictionary
performance_dataframes = {
    'RF': df_performance_RF,
    'SVM': df_performance_SVM,
    'KNN': df_performance_KNN,
    'LR': df_performance_LR,
}

# Write all dataframes to an Excel file
with pd.ExcelWriter('output/performance_summary.xlsx') as writer:
    for model_name, df in performance_dataframes.items():
        df.to_excel(writer, sheet_name=model_name, index=True)


# performance data into single dataframe
df_performance_summary = pd.concat(
    [
        df_performance_RF.loc[['RF (test) - tuned']],
        df_performance_SVM.loc[['SVM (test) - tuned']],
        df_performance_KNN.loc[['KNN (test) - selected & tuned']],
        df_performance_LR.loc[['LR (test) - tuned']],
    ],
    axis=0
)

# performance overview
print("Performance Summary of Models:")
print(df_performance_summary)

# melt dataframe
df_performance_summary = df_performance_summary.reset_index()
df_melted = df_performance_summary.melt(id_vars=['index'], var_name='Metric', value_name='Value')

# filter Dataframes
df_melted_conf_matrix = df_melted[df_melted['Metric'].isin(['tp', 'fp', 'tn', 'fn'])]
df_melted_other_metrics = df_melted[df_melted['Metric'].isin(['accuracy', 'precision', 'recall', 'f1', 'roc_auc'])]

# plot confusion matrix metrics (tp, fp, tn, fn)
plt.figure(figsize=(14, 8))
sns.barplot(x='Metric', y='Value', hue='index', data=df_melted_conf_matrix)
plt.title('Confusion Matrix Metrics Comparison')
plt.ylabel('Score')
plt.xlabel('Metric')
plt.xticks(rotation=45)
plt.legend(title='Model')
plt.tight_layout()
plt.savefig('output/Confusion_Matrix_Metrics_Comparison.png')

# plot metrics (accuracy, recall, precision, f1, roc_auc)
plt.figure(figsize=(14, 8))
sns.barplot(x='Metric', y='Value', hue='index', data=df_melted_other_metrics)
plt.title('Performance Metrics Comparison')
plt.ylabel('Score')
plt.xlabel('Metric')
plt.xticks(rotation=45)
plt.legend(title='Model')
plt.tight_layout()
plt.savefig('output/Performance_Metrics_Comparison.png')

# ROC data
roc_data = {
    'RF selected & tuned (test)': (fp_rates_sel_tune_test, tp_rates_sel_tune_test, df_performance_RF.loc['RF (test) - tuned', 'roc_auc']),
    'SVM (test) - tuned': (fpr_SVM_tune_test, tpr_SVM_tune_test, df_performance_SVM.loc['SVM (test) - tuned', 'roc_auc']),
    'KNN (test) - selected & tuned': (fpr_KNN_tune_test, tpr_KNN_tune_test, df_performance_KNN.loc['KNN (test) - selected & tuned', 'roc_auc']),
    'LR (test) - tuned': (fpr_LR_tune_test, tpr_LR_tune_test, df_performance_LR.loc['LR (test) - tuned', 'roc_auc'])
}

# plot ROC curves
plot_ROC_curve(roc_data, 'ROC_comparison')
