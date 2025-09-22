import numpy as np

NumpyArray = np.ndarray

class LinearInterpolation:
    """This class contains methods for linear interpolation."""
    def __init__(self):
         pass
    
    @classmethod
    def fill_zeros(cls, vec: NumpyArray) -> NumpyArray:
        """Performs the in-place linear interpolation for a vector.

        This method:
        * Returns all zeros if an argument vector has no non-zero entires.
        * Fills leading zeros with the first non-zero value.
        * Fills trailing zeros with the last non-zero value.
        * For any other sequence of zeros interpolates zeros. 
            Interpolation is performed between two non-zero values
            that encompass the sequence of zeros.
        
        Args:
            vec: One-dimensional numpy array.

        Return:
            One-dimensional numpy array with interpolated zero values.

        Example:
            vec:     np.array([0, 1, 0, 3, 4, 0])
            Returns: np.array([1, 1, 2, 3, 4, 4])
        """
        if np.all(vec == 0):
            return vec
        ind = len(vec) - 1
        num_elems = 0
        while vec[ind] == 0.:
            ind -= 1
            num_elems += 1
        if num_elems > 0:
            vec[ind:] = vec[ind]
        ind = 0
        while vec[ind] == 0:
            ind += 1
        vec[:ind] = vec[ind]
        offset = 0
        while ind + offset < len(vec):
            if vec[ind + offset] == 0:
                offset += 1
            elif offset != 0:
                vec[ind - 1 : ind + offset + 1] = np.linspace(
                    vec[ind - 1], vec[ind + offset], offset + 2
                    )
                ind += offset
                offset = 0
            else:
                ind += 1
        return vec

    @classmethod
    def linear_interpolation(cls, data: NumpyArray) -> NumpyArray:
        """Perform the in-place linear interpolation for any numpy array.
        
        This methods reshapes the input `data` into two-dimensional array
        keeping the first dimension unchaged. Then it performs `fill_zeros`, 
        i.e. linear interpolation for any column of the reshaped `data`.

        Args:
            data: Any-dimensional numpy array.

        Returns:
            Any-dimensional numpy array with interpolated zero values.

        Example:
            data: np.array([
                [0, 0, 0, 0],
                [1, 2, 3, 2],
                [0, 0, 0, 0],
                [3, 3, 5, 4],
                [4, 4, 5, 4],
                [0, 0, 1, 1]
                ])
            Returns: np.array([
                [1, 2, 3, 2],
                [1, 2, 3, 2],
                [2, 2, 4, 3],
                [3, 3, 5, 4],
                [4, 4, 5, 4],
                [4, 4, 1, 1]
                ])
        """
        data_shape = data.shape
        if len(data_shape) == 1:
            data = LinearInterpolation.fill_zeros(data)
        else:
            data = np.reshape(data, (data_shape[0], -1))
            data = np.vstack([LinearInterpolation.fill_zeros(vec) for vec in data.T]).T
            data = np.reshape(data, data_shape)
        return data