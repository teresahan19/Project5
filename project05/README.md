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

Please see demonstration.ipynb for the executions and comparisons of our codes. We did three separate implementations after discussing the concept of the algorithm; these are separate scripts in the folder: Implementation 1, Implementation 2, and Implementation 3. The details of each implementation and a reflection detailing our progression from one code to the next is included as an extended "Reflections" section in a Jupyter Notebook entitled "Reflections".

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
Including three unique implementations of the algorithm that illustrates our progression and evolution from simple to complex and from toy to scale was a strategic approach we agreed upon in light of the scaling issues from the previous weeks. Once we had clear sight of how the algorithm should be implemented, we still had time left due to the extended timeframe, and we wanted to see what a scaled implementation would look like. While admittedly, it is far beyond the level of coding most students are at, we felt it would be a good learning tool to look back on.

### Struggles
In the very beginning, we incorrectly implemented the Needleman-Wunsch algorithm, which is for global, not local, alignment. Initialization is very different, as NW progressively penalizes beginning alignment down the sequence the further you go. As such, initialization begins at 0,0 for the first bases, but then goes down by negatives- in our case with a gap penalty of 1- -1, -2, -3, -4, -5, ... for both the initialization row and column. Additionally, NW traceback begins at the bottom right corner, whereas Smith-Waterman takes the global maximum as the start of the alignment. This caused us confusion at first, and we had to identify these differences to understand how each is used. We have included examples of what the matrices would look like for both global and local approaches in our Jupyter notebook for illustration purposes. 

We also struggled in deciding how we should track the traceback. Initially, we thought that the secondary matrix would be unnecessarily taxing on memory to impose, which is why we tried Implementation 1 with a helper function to trace the score directions. Upon testing, however, we realized very quickly that this introduced a bias and was not yielding tracebacks that matched what we were predicting from the score matrix. Additionally, learning how to implement a deterministic tie breaking mechanism was a challenge for us. By breaking ties in preferential order-> diag > up > left, we realized immediately that we were not getting gaps, because the traceback would always choose diagonal. 


### Reflections

Eric Arnold: with the recursive approach, it was good practice to debug the whole thing to see where it was going wrong. For instance, the numerical precision for my score matrix was woefully inadequate (int-8) and only reared its head when trying to align longer sequences and comparing to the other implementations of the function. It was also good practice transferring the array methods into numpy. While pure arrays would be optimal, the function ultimately makes use of a Jit dynamic array (JitList) to store results. The recursive approach only makes sense if you want to get every possible aligned sequence, but it breaks catastrophically when input sequences become complex an can have multiple gaps in different places to get the optimal score, such as with homologous or repeated sections, as discovered in testing.

Stefanie Moreno:
I had a great time working with Thu Thu and Eric on this project. Because I had taken Genomics, I came in with a solid understanding of the biological motivations behind Smith–Waterman, and it was rewarding to be able to share that with the group. What surprised me, though, was how difficult it was at first to translate the conceptual algorithm I understood on paper into clean, executable code. We actually ran through the concepts multiple times by hand before we even started with our pseudocode, so that we could all understand the steps we needed to implement. The template gave us just enough scaffolding to get started, but we still had to make a lot of decisions about how to structure the scoring logic, how to represent traceback, and how to keep the implementation biologically realistic. Working through those decisions together was one of the most valuable parts of the assignment.

One of the most interesting parts of the project was comparing the three implementations. Implementation 1 was simple and easy to read, but it also revealed how subtle mistakes in tie‑breaking or traceback logic can completely distort an alignment. Seeing how diagonal bias or incorrect termination rules produced false motifs or extended alignments into noise helped me appreciate how fragile local alignment can be when the traceback logic isn’t grounded in the score matrix. Implementation 2 fixed those issues and became the first version that behaved like a true Smith–Waterman algorithm. It was deterministic, biologically realistic, and passed all our tests.

At the same time, working with Implementation 2 made its limitations very clear. Even though it was correct, it was still pure Python, and the performance bottlenecks became obvious as soon as we tried longer sequences. Every DP cell update had to go through the Python interpreter, and traceback relied on Python recursion and list operations. It worked beautifully for short sequences, but it simply couldn’t scale. That limitation helped me understand why real bioinformatics tools rarely rely on pure Python for dynamic programming—they need compiled execution to take advantage of hardware‑level optimizations that Python can’t access.

Implementation 3 was our attempt to bridge that gap. Admittedly, this was Eric's implementation, and I am not nearly as advanced in my coding skills. It was fascinating to see how much performance Eric's version could unlock by switching to numeric encoding and using Numba to compile the DP loop and traceback into machine code. Suddenly, the algorithm could handle long sequences and large matrices that would have been impossible in Implementation 2. But the trade‑off was equally clear: the code became harder to read, harder to debug, and much less flexible. Adding new scoring schemes or inspecting intermediate states was no longer straightforward. Running all three implementations side by side made those trade‑offs very concrete. There were even cases where Implementation 2 produced more reliable results than Implementation 3, simply because the compiled version was so much harder to inspect and validate.

Overall, the progression from Implementation 1 to 2 to 3 taught me a lot about the balance between correctness, clarity, and performance. Implementation 1 showed how easy it is to get the logic wrong. Implementation 2 showed how to get the biology right. Implementation 3 showed what it takes to make an algorithm scale to real‑world data. Working through all three gave me a much deeper appreciation for the design decisions behind real alignment tools, and it made the project feel both challenging and genuinely rewarding.

### GenAI appendix
Claude used to generate some of the examples, troubleshoot syntax with numpy
Prompts:
Are there matrices in numpy? How would we go about representing a 2d matrix in numpy?
Generate a series of local alignment tests we can run to evaluate the performance of a smith-waterman dynamic programming script. Include the two sequences to be compared, as well as what the correct alignment should be. Identify what each set of sequences is testing for.
