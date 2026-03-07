import numpy as np
from pprint import pprint
from numba import njit
from numba.typed import List as JitList



decode_map = np.array([' ', 'A', 'C', 'G', 'T', 'N', '-'])




a = "GATNTANNNCA"
b = "TNACAGATTACA"
# a = "ABABAB"
# b = "BABABA"
# a = "ACACNACA"
# b = "ACANCACA"
schema = {"match": 1, "mismatch": -1}


def align_sequences(a, b, schema, seq_a, seq_b, i=0, j=0, cum_score=0, matrix=None):

    if i == len(a) or j == len(b):                      #base case -- we hit the outer edge
        print(seq_a, seq_b, cum_score)
        return
    
    if matrix is None:                                  #init the matrix on first iteration
        matrix = np.zeros(shape=(len(a), len(b)), dtype=np.float16)
    
    if a[i] == b[j]:
        cum_score += schema["match"]
        matrix[i, j] = cum_score
        seq_a += a[i]
        seq_b += b[j]
        align_sequences(a, b, schema, seq_a, seq_b, i=i+1, j=j+1, cum_score=cum_score, matrix=matrix)
    else:
        cum_score += schema["mismatch"]
        if cum_score < 0:
            cum_score = 0
        if cum_score < 0:
            matrix[i, j] = 0
        else:
            matrix[i, j] = 0
        gapped_seq_a = seq_a + "-"
        gapped_seq_b = seq_b + "-"
        align_sequences(a, b, schema, gapped_seq_a, seq_b, i=i, j=j+1, cum_score=cum_score, matrix=matrix)
        align_sequences(a, b, schema, seq_a, gapped_seq_b, i=i+1, j=j, cum_score=cum_score, matrix=matrix)


def fill_matrix(a, b):

    def d(i, j):
        if a[i-1] == b[j-1]:
            return 1
        else:
            return -1
    
    x = len(a) + 1
    y = len(b) + 1
    A = np.zeros(shape=(x, y), dtype=np.int8)

    for i in range(x):
        A[i, 0] = i * -1
    for j in range(y):
        A[0, j] = j * -1

    for i in range(1, x):
        for j in range(1, y):
            score = max([A[i-1, j-1] + d(i, j), A[i-1, j] - 1, A[i, j-1] - 1])
            A[i, j] = score
    
    return A

# def traceback(a, b, i, j, A, B=None):

#     if B is None:
#         B = np.full(shape=A.shape, fill_value=" ", dtype="U10")
    
#     def d(i, j):
#         if a[i-1] == b[j-1]:
#             return 1
#         else:
#             return -1
    
#     B[i, j] = str(A[i, j])
#     print(B)
    
#     if i == 0 and j == 0:
#         return None

#     x = A[i, j]
#     output = []
#     if i > 0:
#         if x == A[i-1, j] + 1 or j == 0:
#             B[i, j] = "u"
#             output.append({"u": traceback(a, b, i-1, j, A, B.copy())})
#     if j > 0:
#         if x == A[i, j-1] + 1 or i == 0:
#             B[i, j] = "l"
#             output.append({"l": traceback(a, b, i, j-1, A, B.copy())})
#     if i > 0 and j > 0:
#         if x == A[i-1, j-1] + d(i, j):
#             B[i, j] = "d"
#             output.append({"d": traceback(a, b, i-1, j-1, A, B.copy())})
#     if len(output) > 0:
#         return output
#     return None


def traceback(a, b, i, j, A, B=None, toprint=False):

    if B is None:
        B = np.full(shape=A.shape, fill_value=" ", dtype="U10")
    
    def d(i, j):
        if a[i-1] == b[j-1]:
            return 1
        else:
            return -1
    if toprint:
        B[i, j] = str(A[i, j])
        print(B)
    
    if i == 0 and j == 0:
        return [{"seq_a": [], "seq_b": []}]

    x = A[i, j]
    output = []
    if i > 0:
        if x == A[i-1, j] - 1 or j == 0:
            B[i, j] = "u"
            next = traceback(a, b, i-1, j, A, B.copy(), toprint=toprint)
            for dct in next:
                dct["seq_a"].append(a[i-1])
                dct["seq_b"].append("-")
            output.extend(next)
    if j > 0:
        if x == A[i, j-1] - 1 or i == 0:
            B[i, j] = "l"
            next = traceback(a, b, i, j-1, A, B.copy(), toprint=toprint)
            for dct in next:
                dct["seq_a"].append("-")
                dct["seq_b"].append(b[j-1])
            output.extend(next)
    if i > 0 and j > 0:
        if x == A[i-1, j-1] + d(i, j):
            B[i, j] = "d"
            next = traceback(a, b, i-1, j-1, A, B.copy(), toprint=toprint)
            for dct in next:
                dct["seq_a"].append(a[i-1])
                dct["seq_b"].append(b[j-1])
            output.extend(next)
    if toprint:
        pprint(output)
        input()

    if len(output) > 0:
        return output
    return [{"seq_a": [], "seq_b": []}]


def init_base_encoding_map():
    '''
    Returns mapping for binary/ASCII encoded strings to numbers
    '''
    base_map = np.zeros(512, dtype=np.uint8)
    base_map[ord('A')] = 1
    base_map[ord('C')] = 2
    base_map[ord('G')] = 3
    base_map[ord('T')] = 4
    base_map[ord('N')] = 5
    base_map[ord('-')] = 6
    return base_map

@njit
def fast_encode_sequences(all_bytes, base_map):
    mdata = np.empty(len(all_bytes), dtype=np.uint8)
    for i in range(len(all_bytes)):
        mdata[i] = base_map[all_bytes[i]]
    return mdata

def encode_seq(seq, base_map=None):
    if base_map is None:
        base_map = init_base_encoding_map()
    
    if type(seq) == str:
        bytes = np.frombuffer(seq.encode(), dtype=np.uint8)
    elif type(seq) == list:
        bytes = np.frombuffer("".join(seq).encode(), dtype=np.uint8)
    
    output = fast_encode_sequences(bytes, base_map)
    return output

def decode_sequence(seq):
    '''        
    :param seq: takes an encoded sequence and converts it back to a string with decode map
    '''
    return "".join(decode_map[seq])

@njit
def fast_traceback(a_encoded, b_encoded, i, j, max_len, A, ptr=0):
    
    def return_empty():
        B = JitList()
        B.append(np.zeros((2, max_len), dtype=np.uint8))
        return B

    def d(i, j):
        if a_encoded[i-1] == b_encoded[j-1]:
            return 1
        else:
            return -1

    if i == 0 and j == 0:
        return return_empty()
    
    P = None
    Q = None
    R = None
    
    output = JitList()
    x = A[i, j]
    if i > 0:
        if x == A[i-1, j] - 1 or j == 0:
            P = fast_traceback(a_encoded, b_encoded, i-1, j, max_len, A, ptr + 1)
            # print("P", P)
            if P is not None:
                for p in range(len(P)):
                    P[p][0][ptr] = a_encoded[i-1]
                    P[p][1][ptr] = 6
                    output.append(P[p])
    if j > 0:
        if x == A[i, j-1] - 1 or i == 0:
            Q = fast_traceback(a_encoded, b_encoded, i, j-1, max_len, A, ptr + 1)
            # print("Q", Q)
            if Q is not None:
                for p in range(len(Q)):
                    Q[p][0][ptr] = 6
                    Q[p][1][ptr] = b_encoded[j-1]
                    output.append(Q[p])
    if i > 0 and j > 0:
        if x == A[i-1, j-1] + d(i, j):
            R = fast_traceback(a_encoded, b_encoded, i-1, j-1, max_len, A, ptr + 1)
            # print("R", R)
            if R is not None:
                for p in range(len(R)):
                    R[p][0][ptr] = a_encoded[i-1]
                    R[p][1][ptr] = b_encoded[j-1]
                    output.append(R[p])

    if len(output) > 0:
        return output
    return return_empty()
    



    


    


    

    
    
    

if __name__ == "__main__":
    # align_sequences(a, b, seq_a="", seq_b="", schema=schema)
    
    A = fill_matrix(a, b)
    print(A)

    print("\npythonic\n")
    paths = traceback(a, b, A.shape[0] - 1, A.shape[1] - 1, A, toprint=False)
    for path in paths:
        for key in path:
            print(key, "".join(path[key]))

    print("\nFAST\n")
    a1 = encode_seq(a)
    b1 = encode_seq(b)
    paths = fast_traceback(a1, 
                           b1, 
                           A.shape[0] - 1, 
                           A.shape[1] - 1, 
                           2 * max(A.shape),
                           A)
    for path in paths:
        for seq in path:
            print(decode_sequence(seq))
    
    seq_b = "ATGGCTAGCTAGCTAGCTAGCTACGATCGATCGATCGATCGNNNATCGATCGATCGTAGCTAGCTAGCTAGCTACGATCGATCGATCGATCG"
    seq_a = "ATGGCTAGCTAGCTAGCTAGCGATCGATCGATCGATCGNNNATCGATCGATCGTAGCTAGCTAGCTAGCTACGATCGATCGATCGATCG"

    import time
    print("----------TESTTEST-----------")

    A = fill_matrix(seq_a, seq_b)

    t0 = time.time()
    paths = traceback(seq_a, seq_b, A.shape[0] - 1, A.shape[1] - 1, A, toprint=False)
    print("TIME", time.time() - t0)
    for path in paths:
        for key in path:
            print(key, "".join(path[key]))

    a1 = encode_seq(seq_a)
    b1 = encode_seq(seq_b)
    t1 = time.time()
    paths = fast_traceback(a1, 
                        b1, 
                        A.shape[0] - 1, 
                        A.shape[1] - 1, 
                        2 * max(A.shape),
                        A)
    print("TIME2", time.time()-t1)
    for path in paths:
        for seq in path:
            print(decode_sequence(seq))

