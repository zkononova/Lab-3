import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt

"""
first part
"""


def svd_custom(a):
    ata = np.dot(a.T, a)
    aat = np.dot(a, a.T)

    eigenvalues_ata, v = np.linalg.eigh(ata)
    eigenvalues_aat, u = np.linalg.eigh(aat)

    # descending order
    sorted_indices_ata = np.argsort(eigenvalues_ata)[::-1]
    sorted_indices_aat = np.argsort(eigenvalues_aat)[::-1]

    v = v[:, sorted_indices_ata]
    u = u[:, sorted_indices_aat]

    singular_values = np.sqrt(eigenvalues_ata[sorted_indices_ata])
    sing_mat = np.diag(singular_values)

    # forming the singular matrix
    sing_mat = np.zeros_like(A, dtype=float)
    min_dim = min(A.shape[0], A.shape[1])
    sing_mat[:min_dim, :min_dim] = np.diag(singular_values[:min_dim])

    for i in range(len(singular_values)):
        if np.linalg.norm(np.dot(A, v[:, i]) - singular_values[i] * u[:, i]) > np.linalg.norm(
                np.dot(A, -v[:, i]) - singular_values[i] * u[:, i]):
            v[:, i] = -v[:, i]

    # verifying the decomposition
    a_reconstructed = np.dot(u, np.dot(sing_mat, v.T))
    return u, sing_mat, v.T, a_reconstructed


# example of the usage
A = np.array([[1, 2], [3, 4], [5, 6]])
u, sing_mat, V_T, A_reconstructed = svd_custom(A)

print("Matrix U:")
print(u)
print("Matrix Σ:")
print(sing_mat)
print("Matrix V^T:")
print(V_T)
print("Reconstructed Matrix A:")
print(A_reconstructed)

assert np.allclose(A, A_reconstructed), "SVD decomposition failed."

"""
second part
"""

file_path = 'ratings.csv'
df = pd.read_csv(file_path)

ratings_matrix = df.pivot(index='userId', columns='movieId', values='rating')

ratings_matrix_filtered = ratings_matrix.dropna(thresh=50, axis=0)  # users who rated 50+ movies
ratings_matrix_filtered = ratings_matrix_filtered.dropna(thresh=10, axis=1)  # movies that have 10+ ratings

ratings_matrix_filled = ratings_matrix_filtered.fillna(2.5)

# from DataFrame to NumPy
R = ratings_matrix_filled.values

user_ratings_mean = np.mean(R, axis=1)

R_demeaned = R - user_ratings_mean.reshape(-1, 1)

# SVD
k = 3
U, sigma, Vt = svds(R_demeaned, k=k)

Sigma = np.diag(sigma)

R_reconstructed = np.dot(U, np.dot(Sigma, Vt)) + user_ratings_mean.reshape(-1, 1)

preds_df = pd.DataFrame(R_reconstructed, columns=ratings_matrix_filtered.columns, index=ratings_matrix_filtered.index)

# Create a new table with only the predicted ratings
predicted_ratings_only = ratings_matrix_filled.copy()
predicted_ratings_only[~ratings_matrix_filtered.isna()] = np.nan
predicted_ratings_only = predicted_ratings_only.fillna(preds_df)

print("Data before prediction: ")
print(ratings_matrix_filled)

print("Data after prediction: ")
print(preds_df)

print("Only predicted data: ")
print(predicted_ratings_only)

print("Matrix U:")
print(U)
print("Matrix Σ:")
print(Sigma)
print("Matrix V^T:")
print(Vt)

# difference between the original matrix and the reconstructed matrix
difference = R_demeaned - (R_reconstructed - user_ratings_mean.reshape(-1, 1))
print("Difference between the original demeaned matrix and the reconstructed matrix:")
print(difference)

print("Max difference:", np.max(np.abs(difference)))

print("Are matrices close?", np.allclose(R_demeaned, R_reconstructed - user_ratings_mean.reshape(-1, 1), atol=1e-5))

"""
users visualisation
"""

U_3d = U[:, :3]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(U_3d[:, 0], U_3d[:, 1], U_3d[:, 2], c='b', marker='o')

ax.set_xlabel('First Dimension')
ax.set_ylabel('Second Dimension')
ax.set_zlabel('Third Dimension')
ax.set_title('Users')

"""
movies visualisation
"""

plt.show()

# i took 3 first columns
V_3d = Vt.T[:, :3]

# i took 20 first movies
num_movies_to_plot = 20
V_3d_selected = V_3d[:num_movies_to_plot, :]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(V_3d_selected[:, 0], V_3d_selected[:, 1], V_3d_selected[:, 2], c='r', marker='o')

ax.set_xlabel('First Dimension')
ax.set_ylabel('Second Dimension')
ax.set_zlabel('Third Dimension')
ax.set_title('Movies')

plt.show()


"""
part with recommendation for each user 
"""
movies_file_path = '/Users/zlata/Documents/GitHub/Lab-3/movies.csv'
movies_df = pd.read_csv(movies_file_path)

print(movies_df.head())


def recommend_movies(user_id, preds_df, movies_df, original_ratings_df, num_recommendations=10):
    user_row_number = user_id - 1
    sorted_user_predictions = preds_df.iloc[user_row_number].sort_values(ascending=False)

    user_data = original_ratings_df[user_row_number]
    user_full = user_data.merge(movies_df, how='left', on='movieId').sort_values(['rating'], ascending=False)

    print(f'User {user_id} has already rated {user_full.shape[0]} movies.')

    recommendations = (movies_df[~movies_df['movieId'].isin(user_full['movieId'])]
                       .merge(pd.DataFrame(sorted_user_predictions).reset_index(), how='left', on='movieId')
                       .rename(columns={user_row_number: 'PredictedRating'})
                       .sort_values(by='PredictedRating', ascending=False)
                       .iloc[:num_recommendations, :-1])

    return user_full, recommendations


user_id = 1
already_rated, predictions = recommend_movies(user_id, preds_df, movies_df, ratings_matrix_filled.values, num_recommendations=10)

