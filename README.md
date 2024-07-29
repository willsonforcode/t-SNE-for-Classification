# t-SNE-for-Classification


# Project Description

This project involves data preprocessing, feature selection, visualization, and filtering of a dataset containing abnormal and normal instances. The primary goal is to identify significant features, visualize the data using t-SNE, and refine the dataset by filtering outliers or less typical data points. The processed data is then saved for further analysis or model development.

## Steps Overview

1. **Data Loading and Merging**:
   - Four CSV files containing abnormal and normal data are loaded.
   - Abnormal and normal datasets are merged into `dfAbnormal_1` and `dfnormal_1`, respectively.
   - A combined dataframe, `df_combined`, is created with labels assigned as 1 (abnormal) and 0 (normal).

2. **Data Scaling and Splitting**:
   - Features (`X`) and labels (`y`) are separated.
   - The features are standardized using `StandardScaler` to ensure that they have a mean of 0 and a standard deviation of 1.
   - The standardized data is split into training and testing sets using `train_test_split`.

3. **Feature Selection Using Logistic Regression**:
   - A logistic regression model with L1 regularization is trained on the training data.
   - Feature weights from the trained model are extracted.
   - The top five features with the highest absolute weights are selected, as these are considered the most significant in distinguishing between normal and abnormal data.

4. **t-SNE Visualization**:
   - t-SNE (t-distributed Stochastic Neighbor Embedding) is applied to reduce the dimensionality of the dataset for visualization.
   - A scatter plot is created, visualizing the t-SNE components, color-coded by the label.

5. **Data Filtering Based on Distance**:
   - The code computes the centers of the classes in the t-SNE space.
   - A filtering process is implemented, where only data points within a specified distance threshold from the class centers are retained. This step helps in focusing on more representative data points and removing potential outliers.

6. **Saving the Filtered Dataset**:
   - Indices of the filtered data points are determined and used to extract the corresponding rows from the original dataset.
   - The final dataset includes the top five selected features and the label.
   - This dataset is saved as a CSV file named "Logistic.csv" for future use.

## Usage

To run this project:

1. Ensure you have the required libraries installed (`pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, etc.).
2. Place the input CSV files (`clean_abnormal1.csv`, `clean_normal.csv`, etc.) in the specified directory.
3. Execute the script to perform the data processing and visualization steps.
4. The final processed data will be saved in "Logistic.csv".

This processed data can be used for further model development, analysis, or as a refined dataset for other machine learning tasks. The visualization and filtering steps are particularly useful for understanding the data distribution and ensuring the quality of the dataset.

---

## Acknowledgements

This project was developed with the assistance of ChatGPT, a language model trained by OpenAI. ChatGPT was used to generate and refine parts of the code, as well as to draft project documentation. The integration of AI tools like ChatGPT helped streamline the coding process and improve the clarity of the project’s documentation.
