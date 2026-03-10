from first_code_runthrough_sm import smith_waterman
from fast_alignment import align_sequences
from pprint import pprint

a = "GATTACAACTG"
b = "GANTCATG"

a = "GATNTANNNCA"
b = "TNACAGATTACA"

a1, b1, matrix = smith_waterman(a, b)
paths = align_sequences(a, b, nw=0)
paths2 = align_sequences(a, b, nw=1)


print(a1)
print(b1)
pprint(paths)
pprint(paths2)

seq_b = "ATGGCTAGCTAGCTAGCTAGCTACGATCGATCGATCGATCGNNNATCGATCGATCGTAGCTAGCTAGCTAGCTACGATCGATCGATCGATCG"
seq_a = "ATGGCTAGCTAGCTAGCTAGCGATCGATCGATCGATCGNNNATCGATCGATCGTAGCTAGCTAGCTAGCTACGATCGATCGATCGATCG"

a1, b1, matrix = smith_waterman(seq_a, seq_b)
paths = align_sequences(seq_a, seq_b, nw=0)
paths2 = align_sequences(seq_a, seq_b, nw=1)


print(a1)
print(b1)
pprint(paths)
pprint(paths2)