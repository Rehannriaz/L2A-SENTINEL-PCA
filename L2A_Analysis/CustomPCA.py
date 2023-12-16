import numpy as np

class CustomPCA:
    def __init__(self, n_components: int, X: np.ndarray) -> None:
        self.n_components = n_components
        self.__X = X
        self.principal_components = None
        self.mean = None
        self.components = None

    def fit_transform(self) -> np.ndarray:
        # Standardize the data
        self.mean = np.mean(self.__X)
        X_standardized = (self.__X - np.mean(self.__X)) / np.std(self.__X)
                
        # Calculate covariance matrix
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
        return self.principal_components

    def inverse_transform(self, principal_components):
        # Reconstruct data from principal components
        reconstructed_data = np.dot(principal_components, self.components.T) + self.mean
        return reconstructed_data
    
    def calculate_error(self) -> float:

        if(self.principal_components == None):
            print("Calculating principal Components...")
            self.inverse_transform(self.__X)

        try:
            X_reconstructed = self.inverse_transform(self.principal_components)
            X_reconstructed = np.transpose(np.reshape(X_reconstructed, (5490, 5490, 10)), axes=(2,0,1))

            concatenated_matrix = np.concatenate(self.__X)

            mse = np.sum((concatenated_matrix - X_reconstructed) ** 2) / concatenated_matrix.size  # Divide by the total number of elements for normalization

            print("No of comps = "+str(self.n_components))
            print("MSE = " + str(mse))

            return mse
        except:
            print("Error calculation failed! please try again...")
            exit(-2)

    
    def __del__(self):
        self.components = None
        self.mean = None
        self.n_components = None