import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, accuracy_score, recall_score, precision_recall_curve, auc, average_precision_score, precision_recall_fscore_support
import xgboost as xgb
from sklearn.pipeline import Pipeline
from typing import Tuple, List

import os
import pickle

ROOT = "C:/Users/mabid/OneDrive/Desktop/Projects - Ongoing/BNP Credit Card Fraud Detection/fraud_detection/"
os.chdir(ROOT)

# If no model is specified, the model with the highest rank will be used

# MODEL_RANK = -1
# EVAL_DATA = os.path.join(ROOT, 'data/processed/model_validation_data.csv')

# # Show models that are in the models directory that has "16" in its name
# active_models = [model for model in os.listdir(os.path.join("models")) if '16' in model]
# print(active_models)
# MODEL_PATH = active_models[MODEL_RANK]

# # Load the model
# with open(os.path.join(ROOT, "models", MODEL_PATH), 'rb') as file:
#     model_args = pickle.load(file)

class Model:
    """
    Class to instantiate a model object :
    Args:
    model_args: dict
        A dictionary containing the following keys:
        - model: The model object
        - pipe: The pipeline object
        - model_name: The name of the model : "string"
        - prediction_matrix_type: The type of prediction matrix used by the model : "string
        - y_true_train: The true labels for the training data : ndarray
        - y_pred_train: The predicted labels for the training data : ndarray
        - y_true_test: The true labels for the testing data : ndarray
        - y_pred_test: The predicted labels for the testing data : ndarray
        
        output = {"model": model,
          "model_params_explicit": params if PARAMS_FILE else None,
          "pipe": pipe,
          "model_name": "XGBoost",
          "prediction_matrix_type": "xgb.DMatrix",
          "y_true_train": y_true_train,
          "y_pred_train": y_pred_train,
          "y_true_test": y_true_test,
          "y_pred_test": y_pred_test,}
    """
    def __init__(self, model_args):
        required_keys = ['model', 'pipe', 'model_name', 'prediction_matrix_type', 
                         'y_true_train', 'y_pred_train', 'y_true_test', 'y_pred_test']
        
        # Check if all required keys are present in model_args
        missing_keys = [key for key in required_keys if key not in model_args]

        if missing_keys:
            raise ValueError(f"Missing required keys in model_args: {missing_keys}")

        # Assigning attributes
        self.model = model_args['model']
        self.pipe = model_args['pipe']
        self.model_name = model_args['model_name']
        self.prediction_matrix_type = model_args['prediction_matrix_type']
        self.y_true_train = model_args['y_true_train']
        self.y_pred_train = model_args['y_pred_train']
        self.y_true_test = model_args['y_true_test']
        self.y_pred_test = model_args['y_pred_test']
        self.y_pred_valid = None
        self.y_true_valid = None

    def load_predict_eval(self, EVAL_DATA):
        eval_data = pd.read_csv(EVAL_DATA)
        X_eval = eval_data.drop('fraud_flag', axis=1)
        y_eval = eval_data['fraud_flag']

        X_eval = self.pipe.transform(X_eval)
        if self.prediction_matrix_type == 'xgb.DMatrix':
            X_eval = xgb.DMatrix(X_eval)

        self.y_pred_valid = self.model.predict(X_eval)
        self.y_true_valid = y_eval


class GenerateReport():
    def __init__(self, model: Model, thresholds: List[float] = [0.01, 0.1]) -> None:
        self.model = model
        self.thresholds = thresholds

    def modified_confusion_matrix(self, true_labels, pred_labels, threshold: float, ax=None, model_name: str = "model") -> None:
        pred_labels = (pred_labels > threshold).astype(int)
        cm = confusion_matrix(pred_labels, true_labels)

        scores_df = pd.DataFrame(precision_recall_fscore_support(pred_labels, true_labels)).T
        scores_df.columns = ["precision", "recall", "fscore", "support"]
        scores_df[['precision', 'recall', 'fscore']] = scores_df[['precision', 'recall', 'fscore']] * 100
        scores_df = scores_df.round({'precision': 2, 'recall': 2, 'fscore': 2})
        overall_accuracy = "{:.2%}".format(accuracy_score(pred_labels, true_labels))

        res = pd.concat([pd.DataFrame(cm), scores_df['recall']], axis=1)
        res = pd.concat([res, scores_df['precision'].to_frame().T], axis=0)

        heatmap = sns.heatmap(res, annot=True, fmt='g', cmap="YlOrRd", cbar_kws={'shrink': 0.8}, ax=ax)

        ax.set_xticklabels([str(i) for i in range(2)] + ["Rate (%)"])
        ax.set_yticklabels([str(i) for i in range(2)] + ["Rate (%)"])
        ax.tick_params(axis='x', which='both', bottom=False, top=True, labelbottom=False, labeltop=True, length=0)

        ax.set_ylabel('Predicted Labels')
        ax.set_title(f"{model_name} Model\nThreshold: {threshold}\nTrue Labels", multialignment='center')
        ax.set_frame_on(False)

    def plot_precision_recall_curve(self, true_labels, pred_labels, ax=None, model_name: str = "model") -> None:
        precision, recall, thresholds = precision_recall_curve(true_labels, pred_labels)
        map = average_precision_score(true_labels, pred_labels)
        
        ax.plot(recall, precision, label='Precision-Recall Curve')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(f'Average Precision Score ({model_name}): {map:.4f}')
        ax.legend()
        ax.plot([0, 1], [1, 0], linestyle='--', label='Baseline')

    def generate_report(self):
        fig, ax = plt.subplots(3, 3, figsize=(24, 18))

        for i, threshold in enumerate(self.thresholds):
            ax_i = ax[0, i]
            self.modified_confusion_matrix(self.model.y_true_train, self.model.y_pred_train, threshold, ax_i, self.model.model_name)
        ax_pr = ax[0, -1]
        self.plot_precision_recall_curve(self.model.y_true_train, self.model.y_pred_train, ax_pr, self.model.model_name)

        for i, threshold in enumerate(self.thresholds):
            ax_i = ax[1, i]
            self.modified_confusion_matrix(self.model.y_true_test, self.model.y_pred_test, threshold, ax_i, self.model.model_name)
        ax_pr = ax[1, -1]
        self.plot_precision_recall_curve(self.model.y_true_test, self.model.y_pred_test, ax_pr, self.model.model_name)

        for i, threshold in enumerate(self.thresholds):
            ax_i = ax[2, i]
            self.modified_confusion_matrix(self.model.y_true_valid, self.model.y_pred_valid, threshold, ax_i, self.model.model_name)
        ax_pr = ax[2, -1]
        self.plot_precision_recall_curve(self.model.y_true_valid, self.model.y_pred_valid, ax_pr, self.model.model_name)
        return ax