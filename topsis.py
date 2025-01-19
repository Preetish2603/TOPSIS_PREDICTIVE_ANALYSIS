import numpy as np
import pandas as pd

def perform_topsis(dataframe, criteria_weights, criteria_impacts):
    def encode_categorical_columns(df):
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype('category').cat.codes
        return df

    def normalize_column(col):
        return col / np.sqrt(np.sum(col ** 2))

    def compute_normalized_matrix(df):
        return df.apply(normalize_column, axis=0)

    def apply_weights_to_matrix(matrix, weights):
        return matrix * weights

    def determine_ideal_solutions(matrix, impacts):
        best_values = []
        worst_values = []
        for index, impact in enumerate(impacts):
            if impact == 'maximize':
                best_values.append(matrix.iloc[:, index].max())
                worst_values.append(matrix.iloc[:, index].min())
            elif impact == 'minimize':
                best_values.append(matrix.iloc[:, index].min())
                worst_values.append(matrix.iloc[:, index].max())
            else:
                raise ValueError("Each impact must be either 'maximize' or 'minimize'.")
        return np.array(best_values), np.array(worst_values)

    def compute_euclidean_distances(matrix, ideal):
        return np.sqrt(np.sum((matrix - ideal) ** 2, axis=1))

    def compute_relative_closeness(d_plus, d_minus):
        return d_minus / (d_plus + d_minus)

    dataframe = encode_categorical_columns(dataframe)
    normalized_matrix = compute_normalized_matrix(dataframe)
    weighted_matrix = apply_weights_to_matrix(normalized_matrix, criteria_weights)
    ideal_best, ideal_worst = determine_ideal_solutions(weighted_matrix, criteria_impacts)
    distances_to_best = compute_euclidean_distances(weighted_matrix.values, ideal_best)
    distances_to_worst = compute_euclidean_distances(weighted_matrix.values, ideal_worst)
    scores = compute_relative_closeness(distances_to_best, distances_to_worst)

    results = dataframe.copy()
    results['TOPSIS Score'] = scores
    results['Rank'] = scores.argsort()[::-1] + 1

    return results
