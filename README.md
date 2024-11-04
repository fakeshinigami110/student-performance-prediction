# K-Nearest Neighbors Classifier


![KNN Classification](images/correlation_heatmap.png)

## K-Nearest Neighbors Classifier

This project implements a KNN classifier using a dataset on mobile device usage and user behavior.

### Project Overview
This project trains a K-Nearest Neighbors model to classify users based on their mobile device behavior. The dataset was sourced from [Kaggle](https://www.kaggle.com/datasets/valakhorasani/mobile-device-usage-and-user-behavior-dataset).

### Code Outline
1. **Dataset Preparation**: Data is loaded, categorical values are encoded, and continuous features are standardized.
2. **Model Training**: A KNN model is trained and evaluated using GridSearchCV to find the best parameters.
3. **Prediction Function**: A function for predicting user class based on input features is provided.
4. **Visualization**: Several visualizations are included to show data distribution and correlations.

### Running the Code
1. Import required libraries:
    ```python
    import pandas as pd
    import numpy as np
    from sklearn.neighbors import KNeighborsClassifier
    ```

2. Preprocess and standardize data:
    ```python
    df = pd.read_csv("user_behavior_dataset.csv")
    df = pd.get_dummies(df, columns=['Device Model'])
    scaler = StandardScaler()
    X_train[continuous_features] = scaler.fit_transform(X_train[continuous_features])
    X_test[continuous_features] = scaler.transform(X_test[continuous_features])
    ```

3. Train and evaluate the model:
    ```python
    knn = KNeighborsClassifier(n_neighbors=5, p=1, weights='distance')
    knn.fit(X_train, y_train)
    ```

4. Run predictions with the custom function:
    ```python
    def predict_user_class(input_features):
        input_features[continuous_features] = scaler.transform(input_features[continuous_features])
        predicted_class = knn.predict(input_features)
        return predicted_class[0]
    ```

### Requirements
- `pandas`
- `numpy`
- `scikit-learn`
- `seaborn`
- `matplotlib`

### Visualizations
- **User Behavior Class by Operating System** ![KNN Classification](images/knn_classification.png)
- **User usage time by OS**![time by os](images/time_by_os_and_class.png)  
- **Average App Usage Time per User Behavior Class** ![Average Usage](images/avrage_usage.png)
- **Mobile Usage Time Distribution by User Class** ![Time Distribution](images/time_distribution_by_class.png)
- **Heatmap of Feature Correlations** ![Heatmap](images/correlation_heatmap.png)

### Results
The trained KNN model achieves a high accuracy in predicting user behavior class, and the visualizations provide insights into device usage patterns among different user classes.

---
