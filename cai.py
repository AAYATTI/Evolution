import numpy as np
from collections import defaultdict
from Bio import SeqIO

# Updated codon frequencies
codon_frequencies = {
    "TTT": 0.00803163, "TTC": 0.00717713, "TTA": 0.00708218, "TTG": 0.00828482,
    "CTT": 0.00569436, "CTC": 0.00508852, "CTA": 0.00502121, "CTG": 0.00587386,
    "ATT": 0.00649090, "ATC": 0.00580032, "ATA": 0.00000000, "ATG": 0.00000000,
    "GTT": 0.00374846, "GTC": 0.00334966, "GTA": 0.00000000, "GTG": 0.00386663,
    "TCT": 0.01510687, "TCC": 0.01349962, "TCA": 0.01332103, "TCG": 0.01558310,
    "CCT": 0.01071064, "CCC": 0.00957111, "CCA": 0.00944449, "CCG": 0.01104828,
    "ACT": 0.01220888, "ACC": 0.01090995, "ACA": 0.01076563, "ACG": 0.01259375,
    "GCT": 0.00705057, "GCC": 0.00630044, "GCA": 0.00621709, "GCG": 0.00727283,
    "TAT": 0.02670698, "TAC": 0.02386556, "TAA": 0.02354985, "TAG": 0.02754888,
    "CAT": 0.01893501, "CAC": 0.01692047, "CAA": 0.01669663, "CAG": 0.01953191,
    "AAT": 0.02158371, "AAC": 0.01928737, "AAA": 0.01903222, "AAG": 0.02226411,
    "GAT": 0.01246448, "GAC": 0.01113835, "GAA": 0.01099101, "GAG": 0.01285740,
    "TGT": 0.03957199, "TGC": 0.03536183, "TGA": 0.03489404, "TGG": 0.04081944,
    "CGT": 0.02805619, "CGC": 0.02507122, "CGA": 0.02473956, "CGG": 0.02894062,
    "AGT": 0.03198079, "AGC": 0.02857828, "AGA": 0.02820023, "AGG": 0.03298894,
    "GGT": 0.01846874, "GGC": 0.01650381, "GGA": 0.01628548, "GGG": 0.01905094,
}

# Amino acid to codon mapping
codon_to_amino_acid = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L", "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M", "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S", "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T", "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "TAT": "Y", "TAC": "Y", "TAA": "Stop", "TAG": "Stop", "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K", "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C", "TGA": "Stop", "TGG": "W", "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R", "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}

# Compute relative adaptiveness (wi) for each codon
relative_adaptiveness = {}
for codon, freq in codon_frequencies.items():
    amino_acid = codon_to_amino_acid.get(codon, None)
    if amino_acid and amino_acid != "Stop":
        max_frequency = max(
            (codon_frequencies[c] for c in codon_to_amino_acid if codon_to_amino_acid[c] == amino_acid),
            default=0
        )
        relative_adaptiveness[codon] = freq / max_frequency if max_frequency else 0

# Function to compute CAI for a gene
def calculate_cai(gene_sequence):
    codons = [gene_sequence[i:i + 3] for i in range(0, len(gene_sequence), 3)]
    wi_values = [relative_adaptiveness.get(codon, min(relative_adaptiveness.values())) for codon in codons]
    if not wi_values or all(w == 0 for w in wi_values):
        return 0.0
    geometric_mean = np.exp(np.mean(np.log([w for w in wi_values if w > 0])))
    return geometric_mean

# Function to read gene sequences from a FASTA file
def read_fasta_sequences(fasta_file):
    sequences = {}
    for record in SeqIO.parse(fasta_file, "fasta"):
        sequences[record.id] = str(record.seq)
    return sequences

# Main code
if __name__ == "__main__":
    fasta_file = "output_dna_uppercase.fasta"
    output_file = "output.log"

    gene_sequences = read_fasta_sequences(fasta_file)
    with open(output_file, "w") as log_file:
        log_file.write("Gene\tCAI\n")
        for gene_id, sequence in gene_sequences.items():
            cai = calculate_cai(sequence)
            log_file.write(f"{gene_id}\t{cai:.5f}\n")

