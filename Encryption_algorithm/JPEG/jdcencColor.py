import numpy as np

def jdcencColor(x, C, k):
    x = int(x)
    
    # Case when x is 0
    if x == 0:
        return 0, np.array([0, 0])  # Return a tuple (0, b) with b as a zero array

    # Determine the category based on x
    category = int(np.floor(np.log2(abs(x)))) + 1

    # Define DC Huffman tables for luma (Y) and chroma (C)
    tabY = np.array([[2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [3, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                     [3, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                     [3, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                     [3, 1, 0, 1, 0, 0, 0, 0, 0, 0],
                     [3, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                     [4, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                     [5, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                     [6, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                     [7, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                     [8, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                     [9, 1, 1, 1, 1, 1, 1, 1, 1, 0]])

    tabC = np.array([[2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [3, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [4, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                     [5, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                     [6, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                     [7, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                     [8, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                     [9, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                     [10, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                     [11, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]])

    b = None  # Placeholder for the output array

    if C == 'Y':
        b = tabY[category, 1:(tabY[category, 0] + 1)]  # Get corresponding Huffman table for luma
    else:
        b = tabC[category, 1:(tabC[category, 0] + 1)]  # Get corresponding Huffman table for chroma

    # Convert x to binary and handle the negative case
    tmp = bin(x)[2:] if x >= 0 else bin(x)[3:]  # Binary string without the negative sign or '0b' prefix
    tmp = [int(i) for i in tmp]
    lls = len(tmp)

    # Ensure k is long enough, pad if necessary
    if len(k) < lls:
        k = [0] * lls  # Initialize k with zeros to the required length

    for m in range(lls):
        # Perform XOR operation between k and tmp
        tmp[m] = 1 if int(k[m]) != tmp[m] else 0

    b = np.append(b, tmp)  # Append the modified tmp to the Huffman table

    b = b.astype(np.int)  # Ensure b is an array of integers
    return lls, b  # Return the length and the final encoded array
