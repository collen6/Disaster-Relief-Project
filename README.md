Project Title: Predictive Modeling for Hurricane Disaster Relief: Search and Rescue Efforts in Haiti
Author: Christian Ollen
Affiliation: Data Science, University of Virginia

Project Overview
This project focuses on developing and applying predictive modeling techniques to support disaster relief efforts in Haiti following the 2010 earthquake. The primary objective is to utilize aerial imagery to identify the presence of blue tarps, which serve as indicators of temporary shelters for displaced persons. This work is aimed at assisting search and rescue teams in efficiently locating individuals in need.

Through the project, various machine learning models were developed, trained, and evaluated on a dataset containing geo-referenced images. The models aim to classify the presence of blue tarps to support targeted relief operations, optimizing sensitivity to ensure maximum detection of affected areas. The models used include both untuned algorithms (Logistic Regression, LDA, QDA) and tuned models (KNN, Penalized Logistic Regression, Random Forest, XGBoost, and SVM with various kernels).

Key Skills Demonstrated
This project showcases critical skills including:

Data Collection and Preprocessing: The project utilizes a large dataset of geo-referenced images, with significant effort dedicated to preprocessing image metadata for use in machine learning models. This includes handling large datasets with over 2 million records for testing and training.

Exploratory Data Analysis (EDA): Key visualizations such as correlation plots, density plots, and class distributions are generated to provide insights into the relationships between RGB color values and the presence of blue tarps. The analysis highlights the data's imbalance and helps refine the modeling process.

Feature Engineering: Data features such as pixel intensity values (Red, Green, Blue) were explored and transformed to optimize model accuracy. Special attention was given to handling multicollinearity among color features.

Modeling and Algorithm Selection: A variety of machine learning algorithms were applied, including:

Logistic Regression
Linear and Quadratic Discriminant Analysis (LDA, QDA)
K-Nearest Neighbors (KNN)
Penalized Logistic Regression (elastic net penalty)
Random Forest (ranger)
XGBoost
Support Vector Machines (SVM) with linear, polynomial, and radial basis function kernels
Hyperparameter Tuning and Cross-Validation: Extensive hyperparameter tuning was performed using grid search to optimize models such as KNN, Random Forest, and SVM. The project employed ten-fold cross-validation to ensure the robustness of model performance.

Model Evaluation and Threshold Optimization: Models were evaluated using metrics such as accuracy, precision, sensitivity, specificity, and ROC AUC. Special attention was given to optimizing sensitivity to ensure the highest possible detection of blue tarps, critical for rescue operations. Threshold selection was fine-tuned to balance the trade-offs between false positives and false negatives.

Visualization and Interpretation of Results: The project includes the generation of ROC curves, performance tables, and visualizations to interpret the model results clearly. These visualizations are critical for communicating findings to non-technical stakeholders involved in disaster response.

Dataset Description
The dataset consists of two parts: training data with approximately 63,241 records and test data with over 2 million records. The key features include:

RGB Pixel Intensity Values: Red, Green, Blue values are used as predictors to identify blue tarps in the images.
Latitude and Longitude: Geo-referenced data that locates the areas of interest.
Class Labels: Labels such as Blue Tarp, Rooftop, Vegetation, and Soil are used for classification.
A key challenge addressed in the project was the class imbalance, with blue tarps only appearing in about 3% of the training data and less than 1% in the test data.

Project Structure
Disaster Relief Project.Rmd: This RMarkdown file contains the code for the entire project, including data preprocessing, EDA, model training, tuning, evaluation, and visualization.
Disaster Relief Project.pdf: A complete report summarizing the methods, results, and conclusions of the analysis, along with visualizations.
Software and Packages Used
The project relies on the following tools and libraries:

RStudio/Posit Cloud: Development environment
tidyverse: Data manipulation and visualization
tidymodels: Machine learning framework for creating models and cross-validation
glmnet: For penalized logistic regression
ggplot2: Data visualization
pROC: ROC curve visualization and comparison
randomForest: Random Forest modeling
xgboost: Boosting algorithm for classification
SVM: Support Vector Machines with different kernels (linear, polynomial, RBF)
kableExtra: For enhanced table formatting
Model Results
Among the models analyzed, Logistic Regression was identified as the most balanced model, performing consistently well across all key metrics. Quadratic Discriminant Analysis (QDA) performed similarly, with slightly higher sensitivity but a higher risk of overfitting.

The final model selected for deployment was Logistic Regression, due to its balance between accuracy, precision, and sensitivity, along with its stability across both the training and holdout datasets.

Conclusion
The predictive models developed in this project can effectively assist in detecting blue tarps, providing critical support for disaster relief efforts in Haiti. By identifying temporary shelters more accurately, the models enable rescue teams to prioritize areas in need, potentially saving lives. The analysis demonstrates how machine learning can be applied to real-world problems with significant humanitarian impact.

Future Work
Further improvements to the model could involve addressing class imbalance through resampling techniques or cost-sensitive learning. Additionally, incorporating advanced image recognition techniques (such as deep learning) could enhance the detection accuracy in future iterations of the project.

References
Gedeck, P. Starting the Disaster Relief Project. virginia.edu.
James, G., Witten, D., Hastie, T., & Tibshirani, R. (2021). An Introduction to Statistical Learning: With Applications in R. Springer US.
Wikimedia Foundation. (2024, June 7). 2010 Haiti earthquake. Wikipedia.
How to Run the Project
Clone the repository: Use the command git clone <repository-url> to clone the project to your local machine.
Install necessary packages: Use RStudio or the terminal to install the required R packages mentioned above.
Run the analysis: Open the Disaster Relief Project.Rmd file in RStudio and run the code step-by-step to see the entire analysis process, from data loading to model evaluation.
View results: The final model's performance and visualizations are available in the Disaster Relief Project.pdf file.
