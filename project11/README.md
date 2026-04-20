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

    let avg_len be the average length of the input sequences


    function _forward_table
        let seq be the input sequence
        let forward matrix be the unfilled matrix
        initialize probability from the initial prob array

        for index in sequence
            get emission column from emission matrix at index

            for state in states:
                previous scores is forward matrix [index - 1]
                transitions is the matrix of transition probabilities at index
                emission is the value in emission column at state

                let scores be prev scores x transitions
                forward matrix [state, index] = product(scores) * emission

    function _backward_table
        let seq be the reversed input sequence sliced by avg_len
        let backward matrix be the unfilled matrix
        initialize first column of backward to be prob = 1

        for index in sequence
            get emission column from emission matrix at avg_len - index

            for state in states:
                previous scores is backward matrix [index - 1]
                transitions is the matrix of transition probabilities at avg_len - index
                scores is previous scores x transitions x emission column
                backward [state, index] = product(scores)

    function baum_welch_profile

        initialize accumulation matrices, num = avg_len
        emission accumulation   (3 x 22 x avg_len)
        transition accumulation (3 x 3 x avg_len)
        initial accumulation

        while not converged do
            for seq in input sequences do
                let gamma be the posterior probability from forward/backward
                let p_seq be the sum of the last column of forward

                add gamma column [0] to initial accumulation

                for index in seq do
                    add gamma column [index] to emission accumulation [index]

                    if index < avg_len - 1 do
                        add to trans accumulation:
                            forward [index] *
                            transitions [index] *
                            emissions [index] *
                            backward [index + 1] /
                            p_seq #to normalize

            normalize accumulations by the sum of accumulated probabilities
            check convergence
            reassign matrices for next iteration
            
                            

            
            
        





    
```

# toy problem:
## execute baum_welch on the sequence

seq = VG--H

### _forward_table

transitions[0] = #our initial state matrix from the stack
|   |     |
|---|-----|
| I | 0.1 |
| M | 0.8 |
| D | 0.0 |
emissions[0] =
|   |  G   |  V   | NOT  |
|---|------|------|------|
| I | 0.80 | 0.20 | 0.00 |
| M | 0.50 | 0.50 | 0.00 |
| D | 0.00 | 0.00 | 0.00 |
      [   ,   ,   ,   ,  ]
fwd = [   ,   ,   ,   ,  ]
      [   ,   ,   ,   ,  ]
  1. For our first emission [V]G--H, multiply the initial state probs by the emission probs. transitions[0] x emissions[0][V]
        [0.1 * 0.2 = .02,   ]
  fwd = [0.8 * 0.5 = .4,   ,]
        [0.0 * 0.0 = 0  ,   ]
  2.
      transitions[1] =
          I   M   D
      I: 0.2 0.7 0.1
      M: 0.3 0.5 0.2
      D: 0.0 0.3 0.7

      emissions[1] =
           G      V    NOT
      I:  0.30  0.50  0.20  
      M:  0.60  0.30  0.10
      D:  0.30  0.40  0.30
  
     For our second emission V[G]--H multiply the previous scores by transition column and the current value of the emission matrix:
    fwd[0][1] = sum(prev_score_col * current_emission_col) * current emission_i_j
          [.02, (.02 * .7 + .4 * .5 + 0 * .3) * 0.3 = .0642]
    fwd = [.08,.4 *   ,]
          [0.0,   ,   ,]
    
### _backward_table
  1.
      transitions[1] =
          I   M   D
      I: 0.2 0.7 0.1
      M: 0.3 0.5 0.2
      D: 0.0 0.3 0.7

      emissions[1] =
           G      V    NOT
      I:  0.30  0.50  0.20  
      M:  0.60  0.30  0.10
      D:  0.30  0.40  0.30
          [   ,   ,   ,   ,  ]
    bwd = [   ,   ,   ,   ,  ]
          [   ,   ,   ,   ,  ]
-

# Reflections

## Eric Arnold

## Thu Thu Han

## Stefanie Moreno

# AI Appendix
