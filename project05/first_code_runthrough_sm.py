import numpy as np

# Direction codes
END, DIAG, UP, LEFT = 0, 1, 2, 3


def cal_score(matrix, seq1, seq2, i, j, match, mismatch, gap):
    """
    Compute score for cell (i,j) and return (score, traceback_direction).

    matrix: numpy 2D scoring matrix
    seq1: vertical sequence (rows)
    seq2: horizontal sequence (cols)
    i,j: current cell indices
    """

    # diag: match/mismatch
    diag = matrix[i-1, j-1] + (match if seq1[i-1] == seq2[j-1] else mismatch)

    # up: gap in seq2
    up = matrix[i-1, j] + gap

    # left: gap in seq1
    left = matrix[i, j-1] + gap

    # local alignment: no negative scores
    score = max(0, diag, up, left)

    # determine traceback direction
    if score == 0:
        direction = END
    elif score == diag:
        direction = DIAG
    elif score == up:
        direction = UP
    else:
        direction = LEFT

    return score, direction


def traceback(seq1, seq2, traceback_matrix, start_pos):
    """
    Reconstruct optimal local alignment from traceback matrix.

    seq1: vertical sequence
    seq2: horizontal sequence
    start_pos: (i,j) where the max score occurs
    """

    # Initialize lists for each aligned sequence (str)
    aligned1 = []
    aligned2 = []

    # Set i, j as starting position in matrix
    i, j = start_pos

    while True:
        move = traceback_matrix[i, j]

        if move == END:
            break

        if move == DIAG:
            aligned1.append(seq1[i-1])
            aligned2.append(seq2[j-1])
            i -= 1
            j -= 1

        elif move == UP:
            aligned1.append(seq1[i-1])
            aligned2.append('-')
            i -= 1

        elif move == LEFT:
            aligned1.append('-')
            aligned2.append(seq2[j-1])
            j -= 1

    # reverse because we built alignment backwards
    return ''.join(reversed(aligned1)), ''.join(reversed(aligned2))


def smith_waterman(seq1, seq2, match=1, mismatch=-1, gap=-1):
    """
    Smith–Waterman local alignment using NumPy array.
    Parameters:
        seq1: (str): string sequence of bases to align
        seq2: (str): string sequence of bases to align
        match: (int): score reward for matching alignments
        mismatch: (int): score penalty for mismatching alignments
        gap (int): score penalty for sequence of unequal lengths
    Returns:
        aligned seq1: (str): string sequence of bases from seq1 with
                             highest alignment score
        aligned seq2: (str): string sequence of bases from seq2 with
                             highest alignment score
        score matrix: (array): alignment score matrix
    """

    # matrix dimensions: (len(seq1)+1) x (len(seq2)+1) -> add 1 for initialization row/col
    rows, cols = len(seq1) + 1, len(seq2) + 1

    # Initialize scoring and traceback matrices with zeros
    score_matrix = np.zeros((rows, cols), dtype=int)
    traceback_matrix = np.zeros((rows, cols), dtype=int)

    # Initialize max score and matrix position variables
    max_score = 0  # Integer
    max_pos = (0, 0)  # Tuple of integers

    # Construct DP matrix
    for i in range(1, rows):
        for j in range(1, cols):
            score, direction = cal_score(score_matrix, seq1, seq2, i, j,
                                         match, mismatch, gap)
            score_matrix[i, j] = score
            traceback_matrix[i, j] = direction

            # Track global maximum for local alignment
            if score > max_score:
                max_score = score
                max_pos = (i, j)

    # Traceback from highest scoring cell
    aligned1, aligned2 = traceback(seq1, seq2, traceback_matrix, max_pos)

    return aligned1, aligned2, score_matrix
