import os
import pandas as pd
import numpy as np
import argparse
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr


def read_csv(file):
    """
    Read a CSV file and convert it into a dictionary format.

    Each row is mapped to a key using the 'id' column. This allows
    for efficient look-up of corresponding ground truth and prediction entries.

    Args:
        file (str): Path to the CSV file.

    Returns:
        dict: A dictionary where keys are IDs and values are the row data.
    """
    df = pd.read_csv(file)
    return {row["id"]: row for _, row in df.iterrows()}


def evaluate(gt_data, pred_data, cols, debug=False):
    """
    Evaluate prediction performance on specified columns.

    Computes several regression metrics:
        - Mean Absolute Error (MAE)
        - Mean Squared Error (MSE)
        - R-squared (R²)
        - Pearson correlation coefficient (R) and its p-value (R_P)

    Args:
        gt_data (dict): Ground truth data loaded from CSV.
        pred_data (dict): Prediction data loaded from CSV.
        cols (list of str): List of columns to evaluate.
        debug (bool): Whether to print raw values for each sample.

    Returns:
        dict: A dictionary mapping each evaluated column to its metrics.
    """
    result = dict()

    for col in cols:
        print(f"\nEvaluating column: {col}")
        gt, pred = [], []

        for key in pred_data.keys():
            gt.append(gt_data[key][col])
            pred.append(pred_data[key][col])

        gt = np.array(gt)
        pred = np.array(pred)

        if debug:
            for i in range(len(gt)):
                print(f"GT: {gt[i]:.3f}, Pred: {pred[i]:.3f}")

        # Compute evaluation metrics
        mae = mean_absolute_error(gt, pred)
        mse = mean_squared_error(gt, pred)
        r2 = r2_score(gt, pred)
        r, r_p = pearsonr(gt, pred)

        result[col] = {
            "MAE": round(mae, 4),
            "MSE": round(mse, 4),
            "R2": round(r2, 4),
            "R": round(r, 4),
            "R_P": round(r_p, 4)
        }

    return result


def main():
    """
    Entry point for the script.

    Uses argparse to load file paths and evaluation settings from the command line.
    Evaluates predictions and prints out metrics for each specified column.
    """
    parser = argparse.ArgumentParser(description="Evaluate prediction results against ground truth ratings.")
    parser.add_argument("--gt_file", type=str, required=True, help="Path to the ground truth CSV file.")
    parser.add_argument("--pred_file", type=str, required=True, help="Path to the prediction CSV file.")
    parser.add_argument("--cols", type=str, required=True,
                        help="Comma-separated list of column names to evaluate (e.g., 'col1,col2,col3').")
    parser.add_argument("--debug", action="store_true", help="Enable to print raw ground truth and prediction values.")

    args = parser.parse_args()

    # Load ground truth and prediction data
    gt_data = read_csv(args.gt_file)
    pred_data = read_csv(args.pred_file)
    cols = [col.strip() for col in args.cols.split(",")]

    # Perform evaluation
    result = evaluate(gt_data, pred_data, cols, debug=args.debug)

    # Display results
    print("\nFinal Evaluation Results:")
    for col, metrics in result.items():
        print(f"{col}: {metrics}")


if __name__ == '__main__':
    main()
