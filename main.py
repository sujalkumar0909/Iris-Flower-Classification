import numpy as np
from scripts.preprocess import load_data, preprocess_data
from scripts.train import train_model
from scripts.evaluate import evaluate
from scripts.feature_importance import plot_feature_importance

if __name__ == "__main__":
    df, iris = load_data()
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
    
    # Train the model
    model = train_model(X_train, y_train)
    
    # Evaluate
    evaluate(model, X_test, y_test, iris.target_names)
    
    # Feature importance
    feature_names = np.array(iris.feature_names)
    plot_feature_importance(model, feature_names)