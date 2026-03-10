
import numpy as np


def initialize_scoring_matrix(seq1, seq2):
    """
    Function to initialize the scoring matrix with 0s for row and column
    seq1: the first sequence to align
    seq2: the second sequence to align
    return: matrix with 0s for row first and 0s for first column
    """
    rows = len(seq1)+1
    cols = len(seq2)+1
    matrix = np.zeros((rows, cols), dtype= int )
    return matrix

def cal_score(matrix, seq1, seq2, i, j, match, mismatch, gap):
    """
    Function to calculate the maximum score for a single cell in the scoring matrix and
    determine the trace back direction that produced that score
    """
    # i is row that represents seq 1
    # j is column that represent seq 2
    #matrix row[1] = seq1[0] first character
    #matrix column[1] = seq2[0] first character

    # Calculating the diagonal score
    # Comparing the two nucleotides from the seq 1 i-1 position and seq 2 j-1 position
    if seq1[i-1] == seq2[j-1]:

        # The two nucleotides match and diagonal score becomes + match score
        diagonal_score = matrix[i-1][j-1] + match

    else:
        # The two nucleotides does not match and diagonal score becomes + mismatch score
        diagonal_score = matrix[i-1][j-1] + mismatch

    # Calculating the up score
    # Come from one row above [i-1] within the same column [j]
    # Adding gap because seq 1 has the character seq 2 does not so gap is inserted in seq 2
    up_score = matrix[i-1][j] + gap

    # Calculating the left score
    # Come from the same row [i] and one column previous [j-1]
    # Adding gap because seq 2 has the character seq 1 does not so gap is inserted in seq 1
    left_score = matrix[i][j-1] + gap

    # Calculates the maximum score to get the final score
    # The 0 in max() indicates that if all scores are negative the final score will be zero
    score = max(0,diagonal_score, up_score, left_score)

    # Call the traceback direction helper to determine which direction the final score came from
    move = __traceback_direction__(score,diagonal_score, up_score, left_score)

    # Return the score which will go to the score matrix and move which will go to the traceback matrix
    return score, move


def __traceback_direction__(score, diagonal_score, up_score, left_score):
    """
    Helper function to calculate the trace back direction
    """
    # This mean all the possible scores is negative
    if score == 0:
        # so the traceback ENDS
        move = 0

    # This mean that the final score came from the diagonal score
    elif score == diagonal_score:
        # so the traceback move diagonally up
        move = 1

    # This mean that the final score came from the score from one row above [i-1] within the same column [j]
    elif score == up_score:
        # so the traceback move up one row same column
        move = 2

    # This mean that the final score came from the score from same row [i] and one column previous [j-1]
    elif score == left_score:
        # so the traceback move back one column same row
        move = 3

    # Unexpected case then END the traceback
    else:
        move = 0

    return move


def traceback(seq1, seq2, traceback_matrix, maximum_position):
    """
    Function to reconstruct the optimal local alignment by following the direction in the traceback matrix
    """
    # Initialize empty strings to store sequences as we traceback
    aligned_seq1 = ""
    aligned_seq2 = ""
    # Initialize the position where the traceback starts
    i , j = maximum_position

    # Follow the traceback matrix till the END 0
    while traceback_matrix[i][j] != 0:

        # Get the direction stored at current cell from __traceback_direction__
        move = traceback_matrix[i][j]

        # Current cell comes from diagonal direction and the nucleotide at seq 1[i-1] and seq 2[j-1] aligns
        if move == 1:
            # add seq1 aligned nucleotide to seq1
            aligned_seq1 = seq1[i-1] + aligned_seq1
            # add seq2 aligned nucleotide to seq2
            aligned_seq2 = seq2[j-1] + aligned_seq2

            # move up diagonally to get the next traceback position
            i = i - 1 # up one row
            j = j -1 # back one column

        # UP gap in seq2
        # Current cell comes from up score
        # seq 1 has one nucleotide seq 2 does not
        elif move == 2:
            # add seq 1 nucleotide to seq 1
            aligned_seq1 = seq1[i-1] + aligned_seq1
            # insert gap to seq 2
            aligned_seq2 = "-" + aligned_seq2
            i = i- 1 #  up one row

        # Left gap in seq1
        # Current cell comes from left score
        # seq 2 has one nucleotide seq 1 does not
        elif move == 3:
            # insert gap to seq 1
            aligned_seq1 = "-"+ aligned_seq1
            # add seq 2 nucleotide to seq 2
            aligned_seq2 = seq2[j-1] + aligned_seq2
            j = j-1 # back one column

    # Return both aligned sequences
    return aligned_seq1, aligned_seq2


def smith_waterman(seq1, seq2, match=1, mismatch=-1, gap=-1):
    """This is the main function to run the smith waterman algorithm in order to find the optimal local alignment"""

    # Initialize the score matrix to store the final score at each cell
    score_matrix     = initialize_scoring_matrix(seq1, seq2)

    # Initialize the traceback matrix to store the direction the final score came from
    traceback_matrix = initialize_scoring_matrix(seq1, seq2)

    # Initialize the maximum tracking variables

    # max score tracks the highest score across all cells in the matrix
    max_score = 0
    # maximum position tracks the i,j from the highest score
    maximum_position = (0, 0)

    # Fill both matrices cell by cell

    # Outer loop row 1 to length of sequence +1 which skips the border row
    for i in range(1, len(seq1) + 1):
        # Inner loop column 1 to length of sequence + 1 which skips the border column
        # Left to right order
        for j in range(1, len(seq2) + 1):

            # Get the calculated scores for both matrices simultaneously
            score_matrix[i][j], traceback_matrix[i][j] = cal_score(
                score_matrix, seq1, seq2, i, j, match, mismatch, gap)

            # Update the max score if the current score is higher
            if score_matrix[i][j] > max_score:
                max_score = score_matrix[i][j]
                maximum_position = (i, j) # store the row and column of the higher score

    # Uses the maximum position through the traceback matrix
    aligned_seq1, aligned_seq2 = traceback(
        seq1, seq2, traceback_matrix, maximum_position)


    return aligned_seq1, aligned_seq2, score_matrix


if __name__ == "__main__":
    seq1 = 'TACTTAG'
    seq2 = 'CACATTAA'

    aligned_seq1, aligned_seq2, score_matrix = smith_waterman(seq1, seq2)
    print(aligned_seq1)
    print(aligned_seq2)
    print(score_matrix)









