fast_alignment.py

import numpy as np
from pprint import pprint
from numba import njit
from numba.typed import List as JitList
import time
import threading
threading.stack_size(64 * 1024 * 1024)


decode_map = np.array([' ', 'A', 'C', 'G', 'T', 'N', '-'])




a = "GATNTANNNCA"
b = "TNACAGATTACA"
# a = "ABABAB"
# b = "BABABA"
# a = "ACACNACA"
# b = "ACANCACA"
schema = {"match": 1, "mismatch": -1}


def align_sequences(a, b, schema, seq_a, seq_b, i=0, j=0, cum_score=0, matrix=None):

    '''
    This is a standard recursive appraoch to the alignment problem. 
    It brutally calls all possible paths using the same recursive structure as NW or SW.
    Not very efficient and this implementation doesn't work properly. Not sure why.
    Don't pay attention to this one.
    '''

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

@njit
def fill_matrix_fast(a, b, nw=1):

    '''
    Fast initalization function for either NW or SW. Uses the respective scoring scheme for either one.
    One weakness of this algorithm is the inability to input special scoring schemes for particular patterns.
    This feature would need to be added in post.
    
    The function uses numba, which complies low-level C and is faster than Numpy for most operations.
    The issue with numba is that you cannot pass python objects as input, so everything needs to be encoded numerically.

    inputs: a, b - the encoded sequences
            nw: set the initialization pattern to needleman-wunsch or to smith-waterman. The latter is nw=0
    
    returns: a matrix with the initial scores.
    '''

    def d(i, j):
        if a[i-1] == b[j-1]:
            return 1
        else:
            return -1
    
    x = len(a) + 1
    y = len(b) + 1
    A = np.zeros(shape=(x, y), dtype=np.int64)

    if nw:
        for i in range(x):
            A[i, 0] = i * -1
        for j in range(y):
            A[0, j] = j * -1
        s = 1
    else:
        s = 1

    for i in range(s, x):
        for j in range(s, y):
            if nw:
                score = max([A[i-1, j-1] + d(i, j), A[i-1, j] - 1, A[i, j-1] - 1])
            else:
                score = max([A[i-1, j-1] + d(i, j), A[i-1, j] - 1, A[i, j-1] - 1, 0])
            A[i, j] = score
    
    return A


def traceback(a, b, i, j, A, B=None, toprint=False):
    '''
    This is a first draft of the traceback function. Notice that it is written completely in python.
    It's 400x slower than fast_traceback. This is huge when working with long sequences.
    You can see the logic of the code more clearly here than in the fast function. If you enable toprint, you get the traceback matrix.
    This function returns all possible alignments, not just the best one.

    Every iteration, it splits down three recursive branches simultaneously and checks to see if they are options.
    This is why there are three conditionals in the middle.
    the [{"seq_a": [], "seq_b": []}] data structure allows multiple kinds of sequences to be returned and keeps them aligned.
    the base-case for this algorithm if i == 0 and j == 0 is specific to needleman-wunsch (global alignment). For smith-waterman, you'd want something else.

    input: a, b - the sequences to be aligned
           i, j - the starting indices for traceback (can be configured for smith waterman)
           A - the score matrix
           B - the matrix that keeps track of the moves; leave none
           toprint - a little variable that prints the traceback matrix as we go
    returns: a list of dicts that represent the possible aligned sequences.
    '''

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

    if len(output) > 0:
        return output
    return [{"seq_a": [], "seq_b": []}]


def init_base_encoding_map():
    '''
    Returns mapping for binary/ASCII encoded strings to numbers
    Note that we have more options compared to Gibbs because we also need to encode -. I added N as well
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
    '''
    njit for fast conversion of sequences to numbers
    '''
    mdata = np.empty(len(all_bytes), dtype=np.uint8)
    for i in range(len(all_bytes)):
        mdata[i] = base_map[all_bytes[i]]
    return mdata

def encode_seq(seq, base_map=None):

    '''
    wrapper that coordinates and calls the encoding function. Handles string or list input.
    '''
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
    decodes and reverses the sequence because the recursive algorithm actually creates it in reverse.
    Eliminates any leftover whitespace

    :param seq: takes an encoded sequence and converts it back to a string with decode map
    '''
    char_list = decode_map[seq]
    return "".join(char_list)[::-1].replace(" ", "")

@njit
def fast_traceback(a_encoded, b_encoded, i, j, max_len, A, ptr=0, nw=1):

    '''
    This is the next level of the pythonic traceback function with changes adapting it to jit and to handle global and local alignemnt.
    Notice the spearate ptr (pointer) variable and max_len. this is because as we increase recursion depth,
    we need to iterate across our numpy array that's intialized with return_empty(). np.arrays cannot grow or shrink and only support direct assignment.
    A workaround in this function is the use of "JitList." This is a special dynamic array for numba that supports appending,
    but doesn't have the overhead of a python list. This is part of the reason why the the function is 400x faster than pure python.
    A more elegant solution to this problem would be to store everything in flat arrays and use separate pointer arrays to delineate
    where sequences stop and end. This would get slightly more performance than the numba dynamic array, but at a huge increase in complexity.
    So, I just went with the numba "JitList" and called it a day.

    input: a_encoded, b_encoded: your encoded sequences
           i, j: the starting indices for traceback
           max_len: the length of the intialized arrays. This needs to be longer than the longest sequence in case there are a lot of gaps.
                    Though, if there are literally as many gaps as the original sequence, this is probably an edge case and the alignment should be discarded.
           A: your score matrix
           ptr: the index where output array values should be assigned
           nw: parameter for alignment mode
    returns: your locally or globally aligned sequences
    '''
    
    def return_empty():
        B = JitList()
        B.append(np.zeros((2, max_len), dtype=np.uint8))
        return B

    def d(i, j):
        if a_encoded[i-1] == b_encoded[j-1]:
            return 1
        else:
            return -1

    if nw:
        if i == 0 and j == 0:
            return return_empty()
    else:
        if A[i, j] == 0:
            return return_empty()
    
    P = None
    Q = None
    R = None
    
    output = JitList()
    x = A[i, j]
    if i > 0:
        if x == A[i-1, j] - 1:
            P = fast_traceback(a_encoded, b_encoded, i-1, j, max_len, A, ptr + 1, nw)
            # print("P", P)
            if P is not None:
                for p in range(len(P)):
                    P[p][0][ptr] = a_encoded[i-1]
                    P[p][1][ptr] = 6
                    output.append(P[p])
    if j > 0:
        if x == A[i, j-1] - 1:
            Q = fast_traceback(a_encoded, b_encoded, i, j-1, max_len, A, ptr + 1, nw)
            # print("Q", Q)
            if Q is not None:
                for p in range(len(Q)):
                    Q[p][0][ptr] = 6
                    Q[p][1][ptr] = b_encoded[j-1]
                    output.append(Q[p])
    if i > 0 and j > 0:
        if x == A[i-1, j-1] + d(i, j):
            R = fast_traceback(a_encoded, b_encoded, i-1, j-1, max_len, A, ptr + 1, nw)
            # print("R", R)
            if R is not None:
                for p in range(len(R)):
                    R[p][0][ptr] = a_encoded[i-1]
                    R[p][1][ptr] = b_encoded[j-1]
                    output.append(R[p])

    if len(output) > 0:
        return output
    return return_empty()
    


def align_sequences(seq_a, seq_b, nw=1):

    '''
    wrapper function that calls everything and encodes everything.
    encoding sequences here slows things down a lot and ideally all sequences would be encoded when the algorithm is initialized.
    
    input: seq_a, seq_b - your string sequences
           nw - your alignment mode

    returns: decoded sequences that have been aligned
    '''

    if type(seq_a) == str:
        seq_a, seq_b = encode_seq(seq_a), encode_seq(seq_b)

    A = fill_matrix_fast(seq_a, seq_b, nw)
    if not nw:
        i, j = np.unravel_index(np.argmax(A), A.shape)
    else:
        i, j = A.shape[0] - 1, A.shape[1] - 1
    paths = fast_traceback(seq_a, seq_b, i, j, 2 * max(A.shape), A, nw=nw)
    return [decode_sequence(seq) for path in paths for seq in path]


    


    


    

    
    
    

if __name__ == "__main__":

    '''all of this is just fun testing stuff'''

    # align_sequences(a, b, seq_a="", seq_b="", schema=schema)
    
    A = fill_matrix_fast(a, b)
    print(A)

    print("\npythonic\n")
    paths = traceback(a, b, A.shape[0] - 1, A.shape[1] - 1, A, toprint=False)
    for path in paths:
        for key in path:
            print(key, "".join(path[key]))

    print("\nFAST\n")
    paths = align_sequences(a, b, nw=1)
    print(paths)
    
    seq_b = "ATGGCTAGCTAGCTAGCTAGCTACGATCGATCGATCGATCGNNNATCGATCGATCGTAGCTAGCTAGCTAGCTACGATCGATCGATCGATCG"
    seq_a = "ATGGCTAGCTAGCTAGCTAGCGATCGATCGATCGATCGNNNATCGATCGATCGTAGCTAGCTAGCTAGCTACGATCGATCGATCGATCG"

    print("----------TESTTEST-----------")

    A = fill_matrix_fast(seq_a, seq_b)

    t0 = time.time()
    paths = traceback(seq_a, seq_b, A.shape[0] - 1, A.shape[1] - 1, A, toprint=False)
    print("TIME", time.time() - t0)
    for path in paths:
        for key in path:
            print(key, "".join(path[key]))

    
    t1 = time.time()
    paths = align_sequences(seq_a, seq_b, nw=1)
    print("TIME2", time.time()-t1)
    print(paths)

