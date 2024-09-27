import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from feature_eng import feature_preprocessing
from models import train_mlp, train_svm, train_gb, train_rf, train_enhanced_nn
from evaluation import evaluate_model
from sklearn.model_selection import train_test_split

# Load and preprocess data
dataset_path = "C:\\Users\\tahak\\OneDrive\\Desktop\\tweede semester\\python for AI\\project\\Mustafa Taha Karadayi(r0879951-Loan payment project)\\Loan_Default.csv"
df, y = feature_preprocessing(dataset_path,target_variable='Status')

# Get test size from user
while True:
    choice = input("Enter the test size (1-0.2, 2-0.25, 3-0.3): ")
    if choice == '1':
        test_size = 0.2
        break
    elif choice == '2':
        test_size = 0.25
        break
    elif choice == '3':
        test_size = 0.3
        break
    else:
        print("Invalid choice. Please try again.")

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=test_size, random_state=42)

# User Menu
def main_menu():
    print("\n===== Select Algorithm =====")
    print("1. Neural Network (MLP)")
    print("2. Support Vector Machine (SVM)")
    print("3. Gradient Boosting Classifier")
    print("4. Random Forest Classifier")
    print("5. Enhanced Neural Network with Regularization")
    print("0. Exit")
    choice = input("\nEnter your choice: ").strip()
    return choice

def main():
    while True:
        choice = main_menu()
        if choice == '1':
            print("\nTraining Neural Network (MLP)...")
            model = train_mlp(X_train, y_train)
            evaluate_model(model, X_test, y_test)
        elif choice == '2':
            print("\nTraining Support Vector Machine (SVM)...")
            model = train_svm(X_train, y_train)
            evaluate_model(model, X_test, y_test)
        elif choice == '3':
            print("\nTraining Gradient Boosting Classifier...")
            model = train_gb(X_train, y_train)
            evaluate_model(model, X_test, y_test)
        elif choice == '4':
            print("\nTraining Random Forest Classifier...")
            model = train_rf(X_train, y_train)
            evaluate_model(model, X_test, y_test)
        elif choice == '5':
            print("\nTraining Enhanced Neural Network with Regularization...")
            model = train_enhanced_nn(X_train, y_train, X_test, y_test)
            evaluate_model(model, X_test, y_test)
            print("\nPerforming Cross-Validation for Enhanced Neural Network...")
        elif choice == '0':
            print("\nExiting program...")
            break
        else:
            print("\nInvalid choice. Please try again.")

if __name__ == "__main__":
    main()