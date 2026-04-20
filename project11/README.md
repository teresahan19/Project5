# Project 11

# Description


# Pseudocode

```python

class profile-HMM
    N is the number of states (1 = Match, 2 = Insert, 3 = Delete)
    L is the max sequence length
    
    let vocab be the 20 amino acids and "-" and "Z" and "Not" (nothing)   #assuming Z is the non-existing amino acid
    let transitions be a stack of all transition probabilities          #dim: N x N x L
    let emissions matrix be a stack of all emission probabilities       #dim: N x vocab x L

    #Emission matrix example:
    M  0.1  0.23...
    I  0.5, 0.25...
    D  0    0...       1.0
        P    R    Y    Not

    let avg_len be the average length of the input sequences # normailzing the column len of the matrix : avg len = sum of seq leq / total no of seqs

    function _forward_table
        let seq be the input sequence # loop through one seq by one seq
        let forward matrix be the unfilled matrix # initialize the matrix with 0s 
        initialize probability from the initial prob array # get first character from obs seq 

        for index in sequence
            get emission column from emission matrix at index # get emmission prob for first char index 

            for state in states:
                previous scores is forward matrix [index - 1] 
                transitions is the matrix of transition probabilities at index
                emission is the value in emission column at state

                let scores be prev scores x transitions
                forward matrix [state, index] = product(scores) * emission # calculation of forward table

    function _backward_table
        let seq be the reversed input sequence sliced by avg_len # reversing sliced obs
        let backward matrix be the unfilled matrix # initilaize the backward matrix with 0s
        initialize first column of backward to be prob = 1 

        for index in sequence
            get emission column from emission matrix at avg_len - index

            for state in states:
                previous scores is backward matrix [index - 1]
                transitions is the matrix of transition probabilities at avg_len - index
                scores is previous scores x transitions x emission column
                backward [state, index] = product(scores)

    function baum_welch_profile

        set iterations
        set threshold for convergence
        initialize accumulation matrices, num = avg_len 
        

        while not converged and less than set iterations do

            initial accumulation
            emission accumulation   (3 x 22 x avg_len) # accumalation matrix for emissions at each obs in avg_len
            transition accumulation (3 x 3 x avg_len)  # accumalation matrix for transitions at each obs in avg_len

            for seq in input sequences do
                let gamma be the posterior probability from forward/backward
                let p_seq be the sum of the last column of forward
                gamma = fwd + bwd - p_seq
                add gamma column [0] to initial accumulation

                for index in seq range of (avg_len) do
                    # adjust the emissin matrix associtate with current index and current seq
                    add gamma column [index] to emission accumulation [index] 

                    if index < avg_len - 1 do # -1 to avoid index error
                        add to trans accumulation:
                            forward [index] *
                            transitions [index] *
                            emissions [index] *
                            backward [index + 1] /
                            p_seq #to normalize


        normalize accumulations by the sum of accumulated probabilities
        check convergence  
        reassign matrices for next iteration


# MSA
 
for seq in sequences
    initiliaze a sequence matrix [no of seq x avg_len] # empty matrix
    fill the sequence one by one (row by row)

 function assign consensus states to columns in sequence matrix
    consensus states = empty list
    set the gap threshold 0.5 

    for each column j in sequence matrix
        get all residues in column j across sequences
        count the number of gaps '-' in column j
        gap percentage = no of gap / total rows in column j 

        if gap percentage < gap threshold
            append M to consensus states list # Match column
        else
            append I to consesus states list # Insertion column

return consensus states


            
            
        





    
```

# Reflections

## Eric Arnold

## Thu Thu Han

## Stefanie Moreno

# AI Appendix
