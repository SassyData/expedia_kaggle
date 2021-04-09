### Hotel.com Property Ranking Task
Hotels.com travellers provide their trip information, like destination, holiday dates and number of people in the search. The search engine (model) then returns a list of properties, ranked by their relevance to that traveller.

##### To run this code
1. Clone the repository
git clone <repo link>
2. Install all packages from the requirements file (paying particular attention to the sklearn version 0.23.1, as this is a dependancy for imblearn, use for resampling)
pip install -r requirements.txt 
3. Open using Jupyter Notebooks / Lab and your prefered IDE.

##### Files
- binary_classification_pipe.py: runs pre-processing, feature engineering, resampling and model training. Also saves pre-processed data & trained models to disk, to save time later.
- get_results.ipynb: uses saved model to make predictions on test set & returns them in a format compatable with the comprtition rules.
- (directory) models/ : saved (trained) models.
- (directory) data/ : raw & pre-processed & engineered data.
- (directory) exploratory/ : a few WIP files, which have not been used in the final model.
- (directory) submissions/ : predictions on the test set, used in competition entry.

##### Results
Models tested were; random forest, SVM, XGBoost, Linear Regression, KNN, 2 Class Bayes
XGB performed best (broadly). 

Comparison of XGB configurations shown here: 
![xgb_comparison_image](master/exploratory/xgb_comp.png)

Best results were achieved with an XGBoost model, using all existing and newly created features, and without resampling the training dataset to improve the extreme class imbalance. 
FINAL SCORE: 0.46742, ranking 52nd.
