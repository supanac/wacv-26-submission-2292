import numpy as np

NumpyArray = np.ndarray

class Kernel:
    def __init__(self, kernel_type:str, kernel_size:int) -> None:
        self.kernel_type = kernel_type
        self.kernel_size = kernel_size
        self.peak = self.kernel_size // 2 + 1
        self.coeffs = self.get_coeffs()
        
    def __call__(self, arr:NumpyArray) -> NumpyArray:
        length, keypts, num_dims = arr.shape
        smoothed_vec = np.reshape(arr, (length, -1)).T
        smoothed_vec = self.expand_vec(smoothed_vec)
        smoothed_vec = np.reshape(smoothed_vec.T, (length + self.kernel_size - 1, keypts, num_dims))
        smoothed_vec = np.array([list(smoothed_vec[i : i + self.kernel_size]) for i in range(arr.shape[0])])
        smoothed_vec = (np.moveaxis(smoothed_vec, 1, -1) @ self.coeffs)
        return smoothed_vec
    
    def get_coeffs(self) -> NumpyArray:
        if self.kernel_type == "linear":
            coeff = 1 / self.kernel_size
            values = np.ones(self.kernel_size) * coeff
        elif self.kernel_type == "triangular":
            coeff = 1 / (self.peak)
            values = (
                1 - np.abs(np.linspace(-1, 1, self.kernel_size + 2))
                )[1:-1]
            values *= coeff
        elif self.kernel_type == "epanechnikov":
            coeff = self.peak / ((self.peak + 0.5) * (self.peak - 0.5))
            values = (np.linspace(-1, 1, self.kernel_size + 2))[1:-1]
            values = 0.75 * (1 - values**2)
            values *= coeff
        return values
    
    def expand_vec(self, matrix: NumpyArray) -> NumpyArray:
        n, m = matrix.shape
        ret_matrix = np.zeros((n, m + self.kernel_size - 1))
        for i, x in enumerate(matrix):
            z_beg = np.polyfit(np.arange(self.peak), x[:self.peak], 1)
            prefix = np.array(
                [np.poly1d(z_beg)(elem) for elem in range(-self.peak + 1, 0)]
            )
            z_end = np.polyfit(np.arange(self.peak), x[-self.peak:], 1)
            suffix = np.array(
                [
                    np.poly1d(z_end)(elem) 
                    for elem in range(self.peak, self.kernel_size)
                ]
            )
            x = np.concatenate([prefix, x, suffix])
            ret_matrix[i] = x
        return ret_matrix