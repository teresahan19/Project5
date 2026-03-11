### Description

Dynamic programming is a computational method for breaking a complex, recursive task into a series of smaller, overlapping subproblems, solving each once, and storing solutions to build the optimal solution. This approach avoids redundant calculations by storing the results of the subproblems- a concept known as memoization-which turns exponential time complexity into polynomial time. In bioinformatics, dynamic programming is used to identify homology between sequences in pairwise sequence alignment, gene recognition, predicting RNA secondary structure, reconstructing phylogenetic trees, and hundreds of other problems. Even algorithms such as BLAST use dynamic programming principles for fast, sensitive searches. By breaking down large sequence comparison tasks, dynamic programming enables faster, more accurate, and tractable analysis of biological data.

Sequence alignment can either be conducted locally or globally to find an optimal alignment. Global alignment employs the Needleman-Wunsch algorithm, whereas Smith-Waterman is used to identify local optimal maxima. The steps of Smith-Waterman are as follows:
1. Create a 2d matrix with the bases of sequence 2 corresponding to each column of the matrix, and the bases from the sequence 1 corresponding to each row. Add an additional row and column for initialization values.
2. For local alignments, we initialize the first row and column with zeros such that beginning an alignment at any position in either sequence does not incur a penalty
3. We begin by comparing the first base of sequence 1 to the first base of sequence 2. We follow the following scoring profile:
       Take the MAXIMUM of:
           0
           diagonal score + match/mismatch
           Up score + gap penalty
           Left score + gap penalty
   And for the current assignment:
           match reward = +1
           mismatch penalty = -1
           gap penalty = -1
4. We then move to the second base of sequence 2 and compare it to the first base of sequence 1 to get a score for the next column. We continue comparing the fisrt base of sequence 1 to every base in sequence 2 to obtain that row's scores, keeping track of where each score was derived from for our traceback.
       ** Important Notes **
           - If the diagonal score plus the mismatch penalty is the maximum value of the 4 comparisons (max of 0, diag, up, left), we keep the diagonal score and align the mismatched bases in our optimal alignment.
           - If the UP score is the maximum of the 4 comparisons, we consume a letter from sequence 1 and put a gap in sequence 2 at this spot in the alignment.
           - If the left score is the maximum of the 4 comparisons, we consume a letter from sequence 2 and put a gap in sequence 1 at this spot in the alignment.
           - If there is a tie in scores, this means it could have come from either direction and we document both in the traceback matrix, because there is potentially more than 1 optimal alignment possible.
6. Once we have completed the scoring matrix, we identify the max score and begin tracing backwards from there, choosing the next largest value for each move, and identifying where it came from in our traceback matrix to determine whether we implement a mismatch, match, gap sequence 1, or gap sequence 2 in our optimal aligmemnt.
7. Alignment terminates upon reaching a score of zero, or reaching the upper left cell in the scoring matrix (which is initialized to zero).
8. We reverse the order of the bases to obtain our local optimal alignment.

Because dynamic programming has a polynomial runtime at best, optimization is important for increasingly complicated alignments or for virtually anything practical. We pondered this critically while constructing our code, and decided to implement different versions to test the impact of optimization in dynamic programming.  

Please see demonstration.ipynb for the implementation of our code. We did three separate implementations after discussing the concept of the algorithm; these are separate scripts in the folder: Implementation 1, Implementation 2, and Implementation 3. The details of each implementation and a reflection detailing our progression from one code to the next is included in the jupyter notebook entitled, "Reflections". 

### Pseudocode

```python

function initialize_matrix
    set first row and col to zeros
    for 2d matrix, set each base of sequence 1 as a single row in matrix
                   set each base of sequence 2 as a single column in matrix
                   leave first row and column blank
    score using score_matrix function for adjacent cells

function score_matrix
    set variables as integers for each direction from current cell from which score could have come
    set current cell as i, j
    calculate score via the following rules:
    max score = take argmax(
                            0
                            diag + match/mismatch
                            up + gap penalty
                            left + gap penalty
                           )

    and set match score, mismatch penalty, gap penalty as integer parameters
    (for this project, match = +1, mismatch = -1, gap = -1)

function traceback
    start_idx = argmax(score_matrix)
    for cell in adjacent_cells:
        if score matches predicted
          save movement step
          move on to next cell
        elif score == 0:
          break loop
```

### Successes
Three unique implementations of the algorithm.

### Struggles
Squashing bugs; comparing output


### Reflections

Eric Arnold: with the recursive approach, it was good practice to debug the whole thing to see where it was going wrong. For instance, the numerical precision for my score matrix was woefully inadequate (int-8) and only reared its head when trying to align longer sequences and comparing to the other implementations of the function. It was also good practice transferring the array methods into numpy. While pure arrays would be optimal, the function ultimately makes use of a Jit dynamic array (JitList) to store results. The recursive approach only makes sense if you want to get every possible aligned sequence, but it breaks catastrophically when input sequences become complex an can have multiple gaps in different places to get the optimal score, such as with homologous or repeated sections, as discovered in testing.

### GenAI appendix
Used to generate some of the examples, troubleshoot syntax with numpy
