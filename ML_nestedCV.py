###############################################################################
# This code was developed by Dr. Pamela Franco for a project focused on 
# multiple sclerosis using PASAT cognitive test. 
#
# A machine learning pipeline was designed, incorporating 
# Sequential Feature Selection (SFS) for feature selection and Nested Cross
# Validation for robust model evaluation. The regression models used include 
# Random Forest, XGBoost, CatBoost, Ridge, Lasso, Logistic Regression, 
# Gradient Boosting, ElasticNet, K-Nearest Neighbors, Decision Tree, 
# Bayesian Ridge, MLP Regressor, Quantile Regression, AdaBoost, and SVM 
# with an RBF kernel. 
#
# The model evaluation is performed using Nested Cross Validation (5-fold outer,
# 5-fold inner) for hyperparameter tuning and performance estimation, followed
# by final evaluation on a held-out test set (80%-20% split). Calculated MSE, 
# MAE, and R-squared. Visualizations include hierarchical clustering, regression 
# plots, residual plots, and nested CV results. Explainability is evaluated 
# using SHAP (for tree-based models).
#
#   Author:      Dr. Pamela Franco
#   Time-stamp:  2025-10-28
#   E-mail:      pamela.franco@unab.cl / pafranco@uc.cl
###############################################################################

import pandas as pd
import matplotlib.pyplot as plt
import re
from sklearn.model_selection import train_test_split, cross_val_predict, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import QuantileRegressor
from sklearn.svm import SVR
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import seaborn as sns
import os
from sklearn.cluster import AgglomerativeClustering
from matplotlib import cm
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from scipy.cluster.hierarchy import dendrogram, linkage
import shap

###############################################################################
# Load data
path = r'C:\Users\pfran\Desktop\Multiple Scleorosis\PASAT'  
data_filename = 'MS_FA_labels_PASAT.csv'
newData = os.path.join(path, data_filename)  

# Reading the dataset
df = pd.read_csv(newData)
features = df.drop(['Label'], axis=1)
label = df['Label'].values
classes = ['HC_CP','RRMS_CP','RRMS_CI']
df = features.rename(columns = lambda x:re.sub('[^*A-Za-z0-9_ ]+', '', x))
feature_names = list(df.columns)

###############################################################################
# Hierarchical Clustering to visualize correlations between features

# Calculate the correlation matrix
correlation_matrix = features.corr()

# Increase figure size to avoid layout issues
plt.rcParams["figure.autolayout"] = True  # This line is already configuring the layout
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# Set a distance threshold to obtain 5 clusters
distance_threshold = 2  

# Perform hierarchical clustering (Agglomerative Clustering) on the correlation matrix (rows and columns)
model = AgglomerativeClustering(n_clusters=None, linkage='average', distance_threshold=distance_threshold)

# Execute clustering for rows and columns
cluster_cols = model.fit_predict(correlation_matrix.T)  # For the columns
cluster_rows = model.fit_predict(correlation_matrix)  # For the rows

# Generate a custom color palette
n_clusters = len(np.unique(cluster_cols)) 
colors = plt.cm.get_cmap('plasma', n_clusters)  

# Assign colors based on clusters
col_colors = [colors(i) for i in cluster_cols]
row_colors = [colors(i) for i in cluster_rows]

# Calculate the number of features in each cluster
cluster_features = [sum(np.array(cluster_cols) == i) for i in range(n_clusters)]

# Create text for each cluster to display in the top right corner
cluster_texts = [f"Cluster {i+1}: {cluster_features[i]:02d} features" for i in range(n_clusters)]

# Perform linkage using the average linkage method
Z = linkage(correlation_matrix, method='average', metric='euclidean')

# Plot the dendrogram
plt.figure(figsize=(12, 8))
dendrogram(Z, labels=correlation_matrix.columns, color_threshold=distance_threshold)
plt.axhline(y=distance_threshold, color='r', linestyle='--', label=f'Distance Threshold = {distance_threshold}')  
#plt.title('Hierarchical Clustering Dendrogram (Average Linkage)')
plt.xlabel('Features')
plt.ylabel('Distance')
plt.savefig('dendogram.png', format='png')
plt.show()

# Plot the clustermap with colored dendrograms
g = sns.clustermap(correlation_matrix, cmap='Blues',
                   figsize=(16, 16),
                   annot=False,  # Do not show the correlation values in each cell
                   xticklabels=True,  # Show labels on the x-axis
                   yticklabels=True,  # Show labels on the y-axis
                   row_cluster=True,  # Ensure clustering for the rows
                   col_cluster=True,  # Ensure clustering for the columns
                   tree_kws={'linewidths': 2},  # Adjust line thickness for the dendrogram
                   row_colors=row_colors,  # Apply cluster colors to rows
                   col_colors=col_colors)  # Apply cluster colors to columns

# Manually adjust the layout to avoid tight layout issues
g.ax_cbar.set_position((0.85, .02, .03, .18))
g.ax_cbar.set_ylabel('Correlation (R)')

# Add cluster information to the top right corner with a box around it
y_position = 0.94  # Initial position for the text
for i, cluster_text in enumerate(cluster_texts):
    if i == len(cluster_texts) - 1:  # Check if it's the last cluster
        plt.text(0.85, y_position, cluster_text, horizontalalignment='left', verticalalignment='top',
                 transform=g.fig.transFigure, fontsize=8, bbox=dict(facecolor=colors(i), edgecolor='none', boxstyle='square,pad=1'),
                 color='black')  # Last cluster text in black
    else:
        plt.text(0.85, y_position, cluster_text, horizontalalignment='left', verticalalignment='top',
                 transform=g.fig.transFigure, fontsize=8, bbox=dict(facecolor=colors(i), edgecolor='none', boxstyle='square,pad=1'),
                 color='white')  # Other cluster texts in white
    y_position -= 0.02  # Decrease the 'y' position for the next cluster text

# Add a rectangle around the cluster labels
plt.gca().add_patch(plt.Rectangle((0.84, 0.8), 0.15, 0.15, linewidth=2, edgecolor='black', facecolor='none'))

# Save and show the figure
plt.savefig('clustermap.png', format='png')
plt.show()

###############################################################################
# Check for missing values in the features dataset
if features.isnull().sum().any():
    print("Warning: There are missing values in the dataset. Imputing with the mean.")
    features = features.fillna(features.mean())

# Split data into training and test sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(features, label,
                                                    test_size=0.2,  # Changed to 0.2 for 80-20 split
                                                    random_state=42)

# 5-fold Stratified Cross-validation for outer and inner loops
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=92)
inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Initialize Random Forest regressor
rf_regressor = RandomForestRegressor(random_state=42)

# Initialize Sequential Feature Selector (SFS)
sfs = SequentialFeatureSelector(estimator=rf_regressor,
                                direction='forward',
                                scoring='neg_mean_squared_error', cv=inner_cv, n_jobs=-1)
  
# Store MSE and SD for plotting
mse_scores = []
mse_std = []

# Perform Sequential Feature Selection to determine the optimal number of features
for n_features in range(1, X_train.shape[1]):  # Try all feature counts from 1 to total features
    sfs.set_params(n_features_to_select=n_features)
    sfs.fit(X_train, y_train)
    
    # Use the selected features for cross-validation
    selected_X_train = X_train.iloc[:, sfs.get_support()]
    
    # Perform cross-validation for each feature subset
    cv_scores = cross_val_score(rf_regressor, selected_X_train, y_train, cv=inner_cv, scoring='neg_mean_squared_error')
    
    # Calculate mean and standard deviation of MSE
    mean_mse = -np.mean(cv_scores)  # Negate because we want to minimize the error
    std_mse = np.std(cv_scores)  # Standard deviation of MSE
    
    mse_scores.append(mean_mse)
    mse_std.append(std_mse)
    
    print(f"Evaluating {n_features} features, MSE: {mean_mse:.4f} ± {std_mse:.4f}")

x_values = range(1, len(mse_scores) + 1)

# Determine the best number of features based on the lowest MSE
best_n_features = np.argmin(mse_scores) + 1  # Add 1 because index is 0-based
min_mse = mse_scores[best_n_features - 1]  # MSE value for the best number of features
min_mse_std = mse_std[best_n_features - 1]  # Standard deviation for the best number of features

# Plot Number of Features vs. MSE
plt.figure(figsize=(10, 6))
plt.errorbar(x_values, mse_scores, yerr=mse_std, fmt='-o', color='blue', capsize=6, label='MSE with error bars')
plt.scatter([best_n_features], [min_mse], color='red', zorder=5, label=f'Best ({best_n_features} features)')
plt.xlabel('Number of Features', fontsize=12)
plt.ylabel('Mean Squared Error (MSE)', fontsize=12)
plt.title('Feature Selection: MSE vs. Number of Features', fontsize=14)
plt.grid(True, axis='x', linestyle='--', alpha=0.7)
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.xlim(0.5, len(mse_scores) + 1)
plt.legend(loc='best', fontsize=10)
plt.tight_layout()
plt.savefig('mse_vs_features_with_min.png', format='png')
plt.show()

# Output the best number of features
print(f"Best number of features: {best_n_features} with MSE: {min_mse:.4f} ± {min_mse_std:.4f}")

# Now, use the best number of features
sfs.set_params(n_features_to_select=best_n_features)
sfs.fit(X_train, y_train)

# Get the indices of the selected features
selected_features_indices = sfs.get_support()  # Get the selected features mask
selected_features = np.where(selected_features_indices)[0]
selected_feature_names = [feature_names[idx] for idx in selected_features]

# Print selected features
print("Total number of features:", X_train.shape[1])  # Total number of features in X_train
print("Number of selected features:", len(selected_features))  # Number of selected features
print("Selected feature names:", selected_feature_names)

# Train the model with selected features
rf_regressor.fit(X_train.iloc[:, selected_features], y_train)  # Train the model with selected features
feature_importances = rf_regressor.feature_importances_

# Create the feature importance DataFrame and verify
if len(feature_importances) == len(selected_features):
    feature_importance_df = pd.DataFrame({
        'feature': X_train.columns[selected_features],
        'importance': feature_importances
    })
    feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

    # Plot the feature importance
    plt.figure(figsize=(6, 2.2))
    sns.barplot(x='importance', y='feature', data=feature_importance_df, color='blue', width=0.8)
    plt.xlabel('Importance', fontsize=8)
    plt.ylabel('Features', fontsize=8)
    plt.yticks(fontsize=8)
    plt.xticks(fontsize=8)
    plt.grid(True, axis='x', linestyle='--', alpha=0.7)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('feature_importance.png', format='png')
    plt.show()
else:
    print("Error: The selected indices are out of range or don't match the features.")

X_train_selected = X_train.iloc[:, selected_features]
X_test_selected = X_test.iloc[:, selected_features]

# Ridge regression model
ridge_regressor = Ridge(alpha=1.0)  # Regularization strength can be adjusted with alpha
ridge_regressor.fit(X_train, y_train)  # Fit the model on all features (X_train)

# Extract feature importance (absolute value of coefficients)
ridge_feature_importance = np.abs(ridge_regressor.coef_)

# Create a DataFrame for feature importance
ridge_feature_importance_df = pd.DataFrame({
    'feature': X_train.columns,  # Use all feature names
    'importance': ridge_feature_importance
})

# Sort by importance
ridge_feature_importance_df = ridge_feature_importance_df.sort_values(by='importance', ascending=False)

# Plot Ridge feature importance
plt.figure(figsize=(24, 10))
sns.barplot(x='feature', y='importance', data=ridge_feature_importance_df, color='blue', width=0.8)
plt.ylabel('Importance', fontsize=16)
plt.xlabel('Features', fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.grid(True, axis='x', linestyle='--', alpha=0.7)
plt.xticks(rotation=90, fontsize=10)
plt.tight_layout()
plt.savefig('ridge_feature_importance.png', format='png')
plt.show()

###############################################################################
###############################################################################
# Nested Cross Validation Implementation

print("Starting Nested Cross Validation...")

# Define models and their parameter grids
models = {
    "Random Forest": {
        'model': RandomForestRegressor(random_state=42),
        'params': {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
        }
    },
    "SVM (RBF Kernel)": {
        'model': SVR(kernel='rbf'),
        'params': {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto', 0.01, 0.1],
            'epsilon': [0.1, 0.2, 0.5],
        }
    },
    "Ridge": {
        'model': Ridge(),
        'params': {
            'alpha': [0.1, 1, 10, 100],
        }
    },
    "Lasso": {
        'model': Lasso(),
        'params': {
            'alpha': [0.1, 1, 10, 100],
        }
    },
    "XGBoost": {
        'model': xgb.XGBRegressor(objective='reg:squarederror', random_state=42),
        'params': {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
        }
    },
    "CatBoost": {
        'model': CatBoostRegressor(silent=True),
        'params': {
            'iterations': [500, 1000],
            'learning_rate': [0.01, 0.1],
            'depth': [6, 8, 10],
            'l2_leaf_reg': [3, 5, 10],
        }
    },
    "Logistic Regression": {
        'model': LogisticRegression(),
        'params': {
            'C': [0.1, 1, 10],
            'penalty': ['l2', 'none'],
            'solver': ['lbfgs', 'liblinear'],
            'max_iter': [100, 200],
        }
    },
    "Gradient Boosting": {
        'model': GradientBoostingRegressor(),
        'params': {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5],
        }
    },
    "Elastic Net": {
        'model': ElasticNet(),
        'params': {
            'alpha': [0.1, 0.5, 1.0],
            'l1_ratio': [0.1, 0.5, 0.9],
        }
    },
    "Bayesian Ridge": {
        'model': BayesianRidge(),
        'params': {
            'alpha_1': [1e-6, 1e-3],
            'alpha_2': [1e-6, 1e-3],
        }
    },
    "K-Neighbors Regressor": {
        'model': KNeighborsRegressor(),
        'params': {
            'n_neighbors': [3, 5, 7],
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree'],
        }
    },
    "MLP Regressor": {
        'model': MLPRegressor(),
        'params': {
            'hidden_layer_sizes': [(50,), (100,), (100, 50)],
            'activation': ['relu', 'tanh'],
            'solver': ['adam', 'lbfgs'],
        }
    },
    "Quantile Regressor": {
        'model': QuantileRegressor(),
        'params': {
            'alpha': [0.1, 0.5, 1.0],
        }
    },
    "Decision Tree": {
        'model': DecisionTreeRegressor(random_state=42),
        'params': {
            'max_depth': [3, 5, 10],
            'min_samples_split': [2, 5],
        }
    },
    "AdaBoost": {
        'model': AdaBoostRegressor(random_state=42),
        'params': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 1],
        }
    }
}

# Store results from nested CV
nested_cv_results = {
    'model': [],
    'outer_fold': [],
    'mse': [],
    'mae': [],
    'r2': []
}

# Store best models from each outer fold
best_models_per_fold = {}

print("Performing Nested Cross Validation...")

# Nested Cross Validation
for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X_train_selected, y_train)):
    print(f"Processing outer fold {fold_idx + 1}/{outer_cv.get_n_splits()}")
    
    # Split data for current outer fold
    X_outer_train, X_outer_test = X_train_selected.iloc[train_idx], X_train_selected.iloc[test_idx]
    y_outer_train, y_outer_test = y_train[train_idx], y_train[test_idx]
    
    best_models_per_fold[fold_idx] = {}
    
    for model_name, model_info in models.items():
        print(f"  Tuning {model_name}...")
        
        # Inner CV for hyperparameter tuning
        grid_search = GridSearchCV(
            estimator=model_info['model'],
            param_grid=model_info['params'],
            cv=inner_cv,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        
        # Fit on inner training data
        grid_search.fit(X_outer_train, y_outer_train)
        
        # Get best model
        best_model = grid_search.best_estimator_
        best_models_per_fold[fold_idx][model_name] = best_model
        
        # Evaluate on outer test fold
        y_pred = best_model.predict(X_outer_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_outer_test, y_pred)
        mae = mean_absolute_error(y_outer_test, y_pred)
        r2 = r2_score(y_outer_test, y_pred)
        
        # Store results
        nested_cv_results['model'].append(model_name)
        nested_cv_results['outer_fold'].append(fold_idx)
        nested_cv_results['mse'].append(mse)
        nested_cv_results['mae'].append(mae)
        nested_cv_results['r2'].append(r2)
        
        print(f"    {model_name}: MSE={mse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")

# Convert results to DataFrame
nested_cv_df = pd.DataFrame(nested_cv_results)

# Calculate mean performance across outer folds for each model
model_performance = nested_cv_df.groupby('model').agg({
    'mse': ['mean', 'std'],
    'mae': ['mean', 'std'],
    'r2': ['mean', 'std']
}).round(4)

print("\nNested Cross Validation Results:")
print("="*50)
for model_name in models.keys():
    model_data = nested_cv_df[nested_cv_df['model'] == model_name]
    if len(model_data) > 0:
        mse_mean = model_data['mse'].mean()
        mse_std = model_data['mse'].std()
        mae_mean = model_data['mae'].mean()
        mae_std = model_data['mae'].std()
        r2_mean = model_data['r2'].mean()
        r2_std = model_data['r2'].std()
        
        print(f"{model_name}:")
        print(f"  MSE: {mse_mean:.4f} ± {mse_std:.4f}")
        print(f"  MAE: {mae_mean:.4f} ± {mae_std:.4f}")
        print(f"  R²:  {r2_mean:.4f} ± {r2_std:.4f}")
        print()

# Find best overall model based on nested CV
best_overall_model_name = nested_cv_df.groupby('model')['mse'].mean().idxmin()
print(f"Best overall model from nested CV: {best_overall_model_name}")

# Now train the best model on the entire training set with the best hyperparameters
print(f"\nTraining best model ({best_overall_model_name}) on full training set...")

# Get the best model configuration from the most frequent best parameters across folds
best_model_folds = []
for fold_idx in best_models_per_fold:
    if best_overall_model_name in best_models_per_fold[fold_idx]:
        best_model_folds.append(best_models_per_fold[fold_idx][best_overall_model_name])

# Use the model from the first fold as our final model (or you could ensemble them)
final_best_model = best_model_folds[0]

# Retrain on full training set with selected features
final_best_model.fit(X_train_selected, y_train)

# Evaluate on the held-out test set
y_test_pred = final_best_model.predict(X_test_selected)

test_mse = mean_squared_error(y_test, y_test_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f"\nFinal Test Set Performance ({best_overall_model_name}):")
print(f"MSE: {test_mse:.4f}")
print(f"MAE: {test_mae:.4f}")
print(f"R²:  {test_r2:.4f}")

###############################################################################
# Plot Nested CV Results
plt.figure(figsize=(12, 8))

# MSE comparison
plt.subplot(2, 2, 1)
mse_means = [nested_cv_df[nested_cv_df['model'] == model]['mse'].mean() for model in models.keys()]
mse_stds = [nested_cv_df[nested_cv_df['model'] == model]['mse'].std() for model in models.keys()]
plt.bar(range(len(models)), mse_means, yerr=mse_stds, capsize=5, alpha=0.7)
plt.xticks(range(len(models)), list(models.keys()), rotation=90)
plt.title('Nested CV - MSE Comparison')
plt.ylabel('MSE')

# MAE comparison
plt.subplot(2, 2, 2)
mae_means = [nested_cv_df[nested_cv_df['model'] == model]['mae'].mean() for model in models.keys()]
mae_stds = [nested_cv_df[nested_cv_df['model'] == model]['mae'].std() for model in models.keys()]
plt.bar(range(len(models)), mae_means, yerr=mae_stds, capsize=5, alpha=0.7, color='orange')
plt.xticks(range(len(models)), list(models.keys()), rotation=90)
plt.title('Nested CV - MAE Comparison')
plt.ylabel('MAE')

# R² comparison
plt.subplot(2, 2, 3)
r2_means = [nested_cv_df[nested_cv_df['model'] == model]['r2'].mean() for model in models.keys()]
r2_stds = [nested_cv_df[nested_cv_df['model'] == model]['r2'].std() for model in models.keys()]
plt.bar(range(len(models)), r2_means, yerr=r2_stds, capsize=5, alpha=0.7, color='green')
plt.xticks(range(len(models)), list(models.keys()), rotation=90)
plt.title('Nested CV - R² Comparison')
plt.ylabel('R²')

plt.tight_layout()
plt.savefig('nested_cv_results.png', format='png')
plt.show()

###############################################################################
# Regression Plot for the best model
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_test_pred, alpha=0.7, color='blue', label='Predictions')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2, label='Perfect Prediction')
plt.xlabel('True Values', fontsize=12)
plt.ylabel('Predictions', fontsize=12)
plt.title(f'Best Model ({best_overall_model_name}) - Test Set\nMSE: {test_mse:.4f}, MAE: {test_mae:.4f}, R²: {test_r2:.4f}', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig('best_model_regression_plot.png', format='png')
plt.show()

###############################################################################
# Residuals Plot for the best model
residuals = y_test - y_test_pred
plt.figure(figsize=(8, 6))
plt.scatter(y_test_pred, residuals, alpha=0.7, color='blue')
plt.axhline(0, color='red', linestyle='--', linewidth=2)
plt.xlabel('Predicted Values', fontsize=12)
plt.ylabel('Residuals', fontsize=12)
plt.title(f'Best Model ({best_overall_model_name}) - Residuals Plot', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('best_model_residuals_plot.png', format='png')
plt.show()

###############################################################################
# SHAP values for the best model if it's a tree-based model
tree_based_models = ["Random Forest", "XGBoost", "CatBoost", "Gradient Boosting", "Decision Tree", "AdaBoost"]

if best_overall_model_name in tree_based_models:
    print(f"\nCalculating SHAP values for {best_overall_model_name}...")
    
    # Initialize SHAP explainer
    explainer = shap.TreeExplainer(final_best_model)
    
    # Calculate SHAP values
    shap_values = explainer.shap_values(X_test_selected)
    
    # SHAP summary plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test_selected, show=False)
    plt.tight_layout()
    plt.savefig('shap_summary_plot_nested_cv.png', format='png')
    plt.show()
    
    # SHAP feature importance
    shap_importance = np.abs(shap_values).mean(0)
    feature_importance_df = pd.DataFrame({
        'feature': X_test_selected.columns,
        'importance': shap_importance
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance_df.head(15))
    plt.title(f'SHAP Feature Importance - {best_overall_model_name}')
    plt.tight_layout()
    plt.savefig('shap_feature_importance_nested_cv.png', format='png')
    plt.show()

print("\nNested Cross Validation completed successfully!")