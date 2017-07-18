# Customer-Prediction
Prediction of customer category in a recharge website to offer appropriate coupons using SVM.
![Alt text](/TrainingDatasetGraph.png?raw=true "Graph formed by training dataset")

Blue Region: Premium Customer
Red Region : Average Customer
Yellow Region: Below Average Customer

Recharge value is recommended on behalf of this prediction.
graph.py is used to generate this graph.
prediction.py is used to make a prediction on behalf of his age, expense, frequency of that expense, and gender.
All these together make the feature vector for the user and is used to train the model.
While prediction, Only age, expense and frequency of expendature is used for prediction.

