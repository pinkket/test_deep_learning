import numpy as np

matrix = np.array([
    [12, 45, 78, 89],
    [56, 78, 90, 123],
    [34, 56, 78, 89],
    [45, 67, 89, 100],
    [23, 45, 67, 78],
    [78, 90, 123, 145],
    [56, 78, 90, 123],
    [34, 56, 78, 89]
])

X_min = np.min(matrix)
X_max = np.max(matrix)
normalized_matrix = (matrix - X_min) / (X_max - X_min)

print(normalized_matrix)
