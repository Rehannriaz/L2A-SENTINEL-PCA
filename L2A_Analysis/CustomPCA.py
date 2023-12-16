import numpy as np

class CustomPCA:
    def __init__(self, n_components: int, X: np.ndarray) -> None:
        self.n_components = n_components
        self.__X = X
        self.principal_components = None
        self.mean = None
        self.components = None
        self.explained_variance = None
        self.explained_variance_ratio = None

    def __standardize_matrix(self, X):
        # Standardize the data
        self.mean = np.mean(self.__X)
        X_standardized = (self.__X - np.mean(self.__X)) / np.std(self.__X)
        return X_standardized                

    def fit_transform(self) -> np.ndarray:
        X_standardized = self.__standardize_matrix(self.__X)

        concat = np.concatenate(X_standardized)
        try:
            covariance_matrix = np.cov(concat, rowvar=False)
        except:
            print("Too much memory being consumed... exiting")
            exit(-1)

        # Calculate eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

        # Sort eigenvalues and corresponding eigenvectors in descending order
        indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[indices]
        eigenvectors = eigenvectors[:, indices]

        # Select the top n_components eigenvectors
        self.components = eigenvectors[:, :self.n_components]

        # Project data onto the selected components
        self.principal_components = np.dot(concat, self.components)
        self.calculate_explained_variance()
        self.calculate_explained_variance_ratio(eigenvalues)

        return self.principal_components

    def inverse_transform(self, principal_components):
        # Reconstruct data from principal components
        reconstructed_data = np.dot(principal_components, self.components.T) + self.mean
        return reconstructed_data
    
    def calculate_error(self) -> float:

        # X = np.transpose(self.__X, axes=(2,1,0))
        X=self.__X
        if not self.principal_components.any():
            print("Calculating principal Components...")
            self.inverse_transform(X)

        try:
            X_reconstructed = self.inverse_transform(self.principal_components)
            X_reconstructed = np.reshape(X_reconstructed, (5490, 5490, 10))

            concatenated_matrix = np.concatenate(X)

            mse = np.sum((self.__standardize_matrix(X) - X_reconstructed) ** 2) / concatenated_matrix.size  # Divide by the total number of elements for normalization

            print("No of components = "+str(self.n_components))
            print("MSE (Error) = " + str(mse))

            return mse
        except Exception as e:
            print("Error calculation failed! please try again...")
            print(e)
            exit(-2)

    def calculate_explained_variance(self):
        # Calculate explained variance for each component
        self.explained_variance = np.var(self.components)

        return self.explained_variance
    
    def calculate_explained_variance_ratio(self, eigenvalues):
        # Calculate explained variance ratio for each component
        self.explained_variance_ratio = eigenvalues / self.explained_variance

        return self.explained_variance_ratio

    
    def __del__(self):
        self.components = None
        self.mean = None
        self.n_components = None
        self.__X = None
        self.principal_components = None