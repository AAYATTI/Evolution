from collections import Counter
from Bio import SeqIO
import sys

def is_parsimony_informative(column):
    """
    A site is parsimony-informative if:
    - At least 2 different nucleotides
    - Each appears at least twice
    """
    counts = Counter(column)
    
    # Remove gaps and Ns
    for bad in ['-', 'N', 'n', '?']:
        counts.pop(bad, None)
    
    # Need at least 2 states with count >= 2
    informative_states = [nt for nt, c in counts.items() if c >= 2]
    
    return len(informative_states) >= 2

def calculate_mutation_rate(fasta_file):
    records = list(SeqIO.parse(fasta_file, "fasta"))
    
    if len(records) < 2:
        raise ValueError("Need at least 2 sequences.")

    alignment_length = len(records[0].seq)
    total_sites = alignment_length
    pi_sites = 0

    for i in range(alignment_length):
        column = [rec.seq[i] for rec in records]
        if is_parsimony_informative(column):
            pi_sites += 1

    mutation_rate = pi_sites / total_sites

    return pi_sites, total_sites, mutation_rate


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python mutation_rate.py alignment.fasta")
        sys.exit(1)

    fasta_file = sys.argv[1]
    pi_sites, total_sites, rate = calculate_mutation_rate(fasta_file)

    print(f"Parsimony-informative sites: {pi_sites}")
    print(f"Total sites: {total_sites}")
    print(f"Mutation rate (PI / total): {rate:.6f}")

