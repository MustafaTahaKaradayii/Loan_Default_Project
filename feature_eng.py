
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, PowerTransformer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import math
from functools import reduce
from sklearn.model_selection import GridSearchCV,KFold
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score

def feature_preprocessing(dataset_path, target_variable):  # load dataset
    df = pd.read_csv(dataset_path)
    print("before applying feature engineering")
    print(f"Dataset shape: {df.shape}")
    # Drop ID and target columns
    df.drop('ID', axis=1, inplace=True)
    y = df[target_variable]
    df.drop(target_variable, axis=1, inplace=True)
    # Identify and drop columns with constant values
    constant_columns = df.columns[df.nunique() == 1]
    if not constant_columns.empty:
        print(f"\nDropping columns with constant values: {constant_columns}")
        df.drop(constant_columns, axis=1, inplace=True)
    
    # Drop the redundant columns to see that two columns are exactly the same
    redundant_cols = []
    for i in range(len(df.columns)):
        col1 = df.iloc[:, i]
        for j in range(i+1, len(df.columns)):
            col2 = df.iloc[:, j]
            if col1.equals(col2):
                redundant_cols.append(df.columns[j])
    
    if redundant_cols:
        print("Dropped redundant columns:", redundant_cols)
        df = df.drop(redundant_cols, axis=1)
    
    """
    commented out because it takes too long
    # Visualize feature distributions
    num_cols = len(df.columns)
    num_rows = math.ceil(num_cols / 4)  # Adjust the number of columns per row
    plt.figure(figsize=(20, 4 * num_rows))
    for i, col in enumerate(df.columns):
        plt.subplot(num_rows, 4, i + 1)
        sns.histplot(df[col], kde=True, element='step')
        plt.title(f'Distribution of {col}')
    plt.tight_layout()
    plt.show()
    """
    
    # Drop some feature according to distribution plot because they have almost same values
    df.drop('Secured_by', axis=1, inplace=True)
    df.drop('Security_Type', axis=1, inplace=True)
    df.drop('construction_type', axis=1, inplace=True)
    df.drop('open_credit',axis=1, inplace=True)

    print(f"Dataset shape: {df.shape}")
    
    # Calculate the correlation matrix for numerical features only
    corr_matrix = df.corr(numeric_only=True)
    highly_corr_features = []
    """" commented out because it takes too long
    # Visualize the Correlation Matrix
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix')
    plt.show()
    """

    
    # Find all columns that are highly correlated with the current column (correlation > 0.8)
    for col in corr_matrix.columns:
        highly_corr_cols = corr_matrix.index[(corr_matrix[col] > 0.8) & (corr_matrix[col] != 1.0)]
        highly_corr_features.extend(highly_corr_cols)
    
    # Remove duplicates from the list of highly correlated features
    highly_corr_features = list(set(highly_corr_features))
    if highly_corr_features:
        print(f"\nRemoving highly correlated features: {highly_corr_features}")
        df.drop(highly_corr_features, axis=1, inplace=True)
    
    """
    # Visualize outliers using box plots for numerical columns
    # commented out because it takes too long
    fig, axes = plt.subplots(nrows=len(df.select_dtypes(include=['float64', 'int64']).columns), ncols=1, figsize=(10, 20))
    for i, col in enumerate(df.select_dtypes(include=['float64', 'int64']).columns):
        sns.boxplot(x=df[col], ax=axes[i])
        axes[i].set_title(f'Box plot of {col}')
    plt.tight_layout()
    plt.show()
    """
    

    # Apply logarithmic transformation
    skewed_features = ['loan_amount', 'property_value', 'income']
    for feature in skewed_features:
        df[feature] = df[feature].astype(float)
        df.loc[:, feature] = np.log1p(df[feature])
    
    # Capping 'Rate of Interest' and 'Interest Rate Spread'
    roi_99th = df['rate_of_interest'].quantile(0.99)
    irs_99th = df['Interest_rate_spread'].quantile(0.99)
    df.loc[df['rate_of_interest'] > roi_99th, 'rate_of_interest'] = roi_99th
    df.loc[df['Interest_rate_spread'] > irs_99th, 'Interest_rate_spread'] = irs_99th

    # Handle outliers using the IQR method
    outlier_indices = []
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        # Use the condition to filter df directly and save indices to filter y later
        condition = ~((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR)))
        outlier_indices.append(df[condition].index)

    # Intersect indices of non-outlier rows across all numerical columns
    if outlier_indices:
        common_indices = reduce(np.intersect1d, outlier_indices)
        df = df.loc[common_indices]
        y = y.loc[common_indices]

    """
    # Visualize missing values
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap="YlGnBu")
    plt.title('Missing Values Heatmap')
    plt.show()
    """
    

    # Impute missing values in numeric and categorical columns
    numeric_features = df.select_dtypes(include=[np.number]).columns
    categorical_features = df.select_dtypes(include=[object, 'category']).columns

    for col in numeric_features:
        if df[col].isnull().sum() > 0:
            median_value = df[col].median()
            df.loc[:, col] = df[col].fillna(median_value)

    for col in categorical_features:
        if df[col].isnull().sum() > 0:
            mode_value = df[col].mode()[0]
            df.loc[:, col] = df[col].fillna(mode_value)
    
    # One-Hot Encoding for 'age' feature and keep a single column
    age_encoder = OneHotEncoder(drop='first', sparse_output=False)
    age_encoded = age_encoder.fit_transform(df[['age']])
    df['age_encoded'] = age_encoded.argmax(axis=1)
    df.drop('age', axis=1, inplace=True)

    # Label Encoding for other categorical features
    label_encoders = {}
    for col in categorical_features:
        if col != 'age':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le

    # Scaling
    scalers = {}
    for col in numeric_features:
        skewness = df[col].skew()
        if abs(skewness) > 1:
            scalers[col] = PowerTransformer(method='yeo-johnson')
            print(f"Applying PowerTransformer (Yeo-Johnson) to {col} due to high skewness ({skewness:.2f})")
        else:
            scalers[col] = StandardScaler()
            print(f"Applying StandardScaler to {col} (skewness: {skewness:.2f})")

    for col, scaler in scalers.items():
        df[col] = scaler.fit_transform(df[[col]])

    # StandardScaler for encoded categorical features
    categorical_scaler = StandardScaler()
    encoded_categorical_features = [col for col in categorical_features if col != 'age']
    df[encoded_categorical_features] = categorical_scaler.fit_transform(df[encoded_categorical_features])
    
    # Step 1: Define the PCA model to retain 95% variance
    pca = PCA(n_components=0.95)  # Adjust n_components as needed for the desired variance retention

    # Step 2: Fit the PCA model and transform the data
    pca_result = pca.fit_transform(df)

    # Step 3: Create a new DataFrame with the PCA results
    df_pca = pd.DataFrame(pca_result, index=df.index)

    # Step 4: Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(df_pca, y, test_size=0.3, random_state=42)

    # Step 5: Initialize and train RandomForestClassifier on the training set
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    # Step 6: Make predictions on the test set
    y_pred = rf.predict(X_test)

    # Step 7: Evaluate the model's performance
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy of RandomForest: {accuracy}")
    print(classification_report(y_test, y_pred))
    
    # Step 8: Identify and drop the least important PCA components
    importances = rf.feature_importances_
    sorted_indices = np.argsort(importances)
    n_drop = 8  # Number of least important PCA components to drop

    # Drop the least important PCA components
    features_to_drop = [df_pca.columns[index] for index in sorted_indices[:n_drop]]
    df_pca.drop(columns=features_to_drop, inplace=True)

    print("after applying feature engineering")
    print(f"Dataset shape: {df_pca.shape}")
    
    # Step 9: Split data for training, validation, and testing
    X_train, X_temp, y_train, y_temp = train_test_split(df_pca, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Print shapes to verify the splits
    print(f"Train set shape: {X_train.shape}")
    print(f"Validation set shape: {X_val.shape}")
    print(f"Test set shape: {X_test.shape}")

   # Step 10: Perform K-fold cross-validation with k=5
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_accuracies = []

    for train_index, test_index in kf.split(df_pca):
        X_train_kf, X_test_kf = df_pca.iloc[train_index], df_pca.iloc[test_index]
        y_train_kf, y_test_kf = y.iloc[train_index], y.iloc[test_index]
        
        rf_kf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_kf.fit(X_train_kf, y_train_kf)
        y_pred_kf = rf_kf.predict(X_test_kf)
        
        accuracy_kf = accuracy_score(y_test_kf, y_pred_kf)
        cv_accuracies.append(accuracy_kf)

    print(f"K-fold cross-validation accuracies: {cv_accuracies}")
    print(f"Mean cross-validation accuracy: {np.mean(cv_accuracies)}")
    
    return df_pca, y









