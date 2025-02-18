import numpy as np
from numpy.typing import NDArray, ArrayLike

def abc(arr: NDArray) -> None:
    print(arr.shape)
    print(arr.ndim)
    print(np.sum(arr, axis=1, keepdims=True))
    temp = np.sum(arr, axis=1, keepdims=True)
    print(arr - temp)
    # print(arr)
    print(np.empty_like(arr))
    aa = arr.reshape(-1, 1)
    bb = arr.reshape(-1, 1).T
    print(aa)
    print(bb)
    print(np.dot(aa, bb))
    print(np.argmax(arr, axis=1, keepdims=False))

if __name__ == "__main__":
    arr = np.array([[-4, 9, -2, 0, 3, 7], [7, 3, 0, -2, 9, -4]])
    abc(arr)
    print(np.empty_like(arr))
    print(1/(1 + np.exp(-arr)))