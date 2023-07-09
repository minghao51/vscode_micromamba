import logging
import sklearn.metrics
import pandas as pd
import shap



def log_shap_plots(model, options, X, top_idx:int = 10):
    """extract top 10 avg shap values feature name"""

def log_shap_plots(model, options, X):
    """plot shap summary, dependence, force and waterfall plots with shap.""" 
    import shap
    explainer = shap.Explainer(automl1.model.estimator, algorithm ='Tree')
    shap_values = explainer(X)

    shap.plots.bar(shap_values)
    shap.summary_plot(model, x, **options)
    shap.dependence_plot("RM", shap_values, x, **options)
    shap.force_plot(explainer.expected_value, shap_values, x, **options)
    shap.waterfall_plot(explainer.expected_value, shap_values[0,:], x.iloc[0,:], **options)

def log_classification_report_metrics(y_true, y_pred, output_format):
    """Log all metrics from classification_report of sklearn.

    Args:
    y_true: The ground truth labels.
    y_pred: The predicted labels.

    Returns:
    A dictionary of metrics.
    """
    report = sklearn.metrics.classification_report(y_true, y_pred, output_dict=True)
    if output_format == 'dict':
        return report
    elif output_format == 'pandas':
        report_df = pd.DataFrame(report).T
        report_df.index.name = "Class"
        report_df.columns = ["Precision", "Recall", "F1-Score", "Support"]
        return report_df

def log_binary_classification_metrics(y_true, y_pred):
    """
    Log all relevant metrics for binary classification.

    Args:
        y_true: The ground truth labels.
        y_pred: The predicted labels.

    Returns:
        A dictionary of metrics.
    """

    metrics = {}
    metrics["accuracy"] = sklearn.metrics.accuracy_score(y_true, y_pred)
    metrics["precision"] = sklearn.metrics.precision_score(y_true, y_pred)
    metrics["recall"] = sklearn.metrics.recall_score(y_true, y_pred)
    metrics["f1_score"] = sklearn.metrics.f1_score(y_true, y_pred)
    metrics["confusion_matrix"] = sklearn.metrics.confusion_matrix(y_true, y_pred)

    logging.info("Binary classification metrics:")
    for metric, value in metrics.items():
        logging.info("  %s: %s", metric, value)

    return metrics

def log_binary_predictions_plots(y_true, y_pred):
    """plot detetion error trade off, roc and apc with sklearn.

    Args:
        y_true: The ground truth labels.
        y_pred: The predicted labels.

    Returns:
        A dictionary of plots.
        """
        detection_error_tradeoff = sklearn.metrics.detetion_error_tradeoff(y_true, y_pred)
        roc = sklearn.metrics.roc_curve(y_true, y_pred)
        apc = sklearn.metrics.average_precision_score(y_true, y_pred)
        logging.info('Detection error tradeoff: %s', detection_error_tradeoff)
        logging.info('ROC: %s', roc)
        logging.info('APC: %s', apc)
        return detection_error_tradeoff, roc, apc
    
from yellowbrick.target import ClassBalance
from yellowbrick.classifier import confusion_matrix, precision_recall_curve, classification_report, discrimination_threshold, roc_auc
def log_yellowbricks_plots(model, x, y):
    """plot class balance, confusion matrix, precision recall curve, classification report, discrimination threshold and roc auc with yellowbricks.

    Args:
        model: The model.
        x: The features.
        y: The labels.

    Returns:
        A dictionary of plots.
        """
        class_balance = ClassBalance(model)
        confusion_matrix = confusion_matrix(model, x, y)
        precision_recall_curve = precision_recall_curve(model, x, y)
        classification_report = classification_report(model, x, y)
        discrimination_threshold = discrimination_threshold(model, x, y)
        roc_auc = roc_auc(model, x, y)
        logging.info('Class balance: %s', class_balance)
        logging.info('Confusion matrix: %s', confusion_matrix)
                     



