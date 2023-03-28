# Model Card

For additional information see the Model Card paper: [https://arxiv.org/pdf/1810.03993.pdf](https://arxiv.org/pdf/1810.03993.pdf)

## Model Details
This model leverages a `RandomForestClassifier` from the `scikit-learn` library for classification tasks. It uses the classifier's default hyperparameters.

## Intended Use
The model is designed to classify employee salaries into two categories, `<=50K` and `>50K`, based on a variety of employee characteristics.

Users can input employee information in the specified format to receive salary category predictions.

## Training Data
The model is trained and evaluated using a publicly available dataset from the Census Bureau. The dataset can be accessed at [https://archive.ics.uci.edu/ml/datasets/census+income](https://archive.ics.uci.edu/ml/datasets/census+income).

The dataset contains a substantial number of examples and a diverse range of features, offering sufficient data to train a high-performing model.

During both training and evaluation, categorical features are encoded with `OneHotEncoder`, and the target is transformed using `LabelBinarizer`.

## Evaluation Data
The original dataset is preprocessed and subsequently split into training and evaluation datasets, with the evaluation dataset constituting 20% of the data.

## Metrics
The model's performance is evaluated using three metrics: precision, recall, and F-beta score. The model achieves the following results:
* Precision: 0.7425
* Recall: 0.6446
* F-beta score: 0.6901

## Ethical Considerations
The misuse of census data could negatively impact the individuals surveyed and possibly others in related populations.

## Caveats and Recommendations
The model's training data primarily features individuals from the USA. Consequently, it is not advisable to use this model for predicting salary categories for people from other regions worldwide, as they may have significantly different feature distributions.
