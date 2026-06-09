#!/usr/bin/env python3

# =============================================================================
# HEPTOSYLTRANSFERASE I VARIANTS – SYNTENY & OPERON ARCHITECTURE PIPELINE
# =============================================================================
#
# Author  : Aayatti Mallick Gupta
#
# -----------------------------------------------------------------------------
# Overview
# -----------------------------------------------------------------------------
# This pipeline reconstructs gene synteny and operon architecture
# surrounding heptosyltransferase I variants:
#
#   • HepI (rfaC / waaC)
#   • OpsX-like systems
#
# -----------------------------------------------------------------------------
# Workflow
# -----------------------------------------------------------------------------
# 1. Identification of target CDS hits from genome assemblies
# 2. Extraction of genomic neighborhood context (± window genes)
# 3. Construction of gene presence matrix across genomes
# 4. Conservation analysis and frequency profiling
# 5. Operon architecture inference
# 6. Visualization of conserved synteny and top operons
#
# -----------------------------------------------------------------------------
# Outputs
# -----------------------------------------------------------------------------
# • protein_hits.tsv
# • gene_neighborhood.tsv
# • gene_frequency.tsv
# • genome_gene_matrix.tsv
# • operon_architecture.tsv
# • gene_neighborhood_heatmap.jpg
# • gene_neighborhood_frequency.jpg
# • top_operons_diagram.svg
#
# =============================================================================


import os
import re
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from dna_features_viewer import GraphicFeature, GraphicRecord


# =============================================================================
# CONFIGURATION
# =============================================================================

TAXONOMY_FILE = "HepI_fasta_taxonomy_with_genome.tsv"
GENOME_BASE_DIR = "/media/aayatti/Seagate Hub/OpsX_vs_HepI/HepI/cdna-retreival/ncbi_dataset/data"

OUTDIR = "synteny_results_local"
os.makedirs(OUTDIR, exist_ok=True)

# Target proteins (HepI + OpsX)
TARGET_PROTEINS = [
    "lipopolysaccharide heptosyltransferase I",
    "lipopolysaccharide heptosyltransferase RfaC",
    "waaC",
    "OpsX"
]

# Neighborhood window (gene count)
WINDOW_GENES = 10

TOP_OPERONS = 10


# =============================================================================
# STEP 1: IDENTIFY TARGET PROTEINS
# =============================================================================

print("\n[STEP 1] Extracting HepI / OpsX protein hits...")

df = pd.read_csv(TAXONOMY_FILE, sep="\t")
df["species_key"] = df["FASTA_header"].str.split("|").str[0]

hits = []
proteins = []

for i, row in df.iterrows():

    species = row["species_key"]
    accession = row["GenomeAccession"]

    fasta_file = os.path.join(
        GENOME_BASE_DIR,
        accession,
        "cds_from_genomic.fna"
    )

    if not os.path.exists(fasta_file):
        continue

    for rec in SeqIO.parse(fasta_file, "fasta"):

        header = rec.description.lower()

        if any(t.lower() in header for t in TARGET_PROTEINS):

            match = next(
                t for t in TARGET_PROTEINS
                if t.lower() in header
            )

            hits.append({
                "Species": species,
                "Genome": accession,
                "CDS_ID": rec.id,
                "Header": rec.description
            })

            proteins.append(
                SeqRecord(
                    Seq(rec.seq),
                    id=f"{accession}_{rec.id}",
                    description=f"{species} | {match}"
                )
            )

pd.DataFrame(hits).to_csv(
    os.path.join(OUTDIR, "protein_hits.tsv"),
    sep="\t",
    index=False
)

SeqIO.write(
    proteins,
    os.path.join(OUTDIR, "hepi_opsx_hits.faa"),
    "fasta"
)

print("[DONE] Protein hits extracted")


# =============================================================================
# STEP 2: GENOMIC NEIGHBORHOOD EXTRACTION
# =============================================================================

print("\n[STEP 2] Extracting gene neighborhoods...")

def parse_location(header):
    m = re.search(r"(\d+)\.\.(\d+)", header)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None


neighborhood = []

for hit in hits:

    genome = hit["Genome"]

    fasta_file = os.path.join(
        GENOME_BASE_DIR,
        genome,
        "cds_from_genomic.fna"
    )

    if not os.path.exists(fasta_file):
        continue

    records = list(SeqIO.parse(fasta_file, "fasta"))

    for i, rec in enumerate(records):

        start, end = parse_location(rec.description)

        if start is None:
            continue

        window = records[
            max(0, i - WINDOW_GENES):
            i + WINDOW_GENES + 1
        ]

        for wrec in window:

            wstart, wend = parse_location(wrec.description)

            if wstart is None:
                continue

            neighborhood.append({
                "Species": hit["Species"],
                "Genome": genome,
                "NeighborGene": wrec.id,
                "Start": wstart,
                "End": wend,
                "Product": wrec.description
            })

neigh_df = pd.DataFrame(neighborhood)

neigh_df.to_csv(
    os.path.join(OUTDIR, "gene_neighborhood.tsv"),
    sep="\t",
    index=False
)

print("[DONE] Gene neighborhoods extracted")


# =============================================================================
# STEP 3: GENE FREQUENCY ANALYSIS
# =============================================================================

print("\n[STEP 3] Gene conservation analysis...")

neigh_df["Gene"] = neigh_df["Product"].fillna("Unknown")

gene_freq = neigh_df.groupby("Gene")["Genome"].nunique()

gene_freq.to_csv(
    os.path.join(OUTDIR, "gene_frequency.tsv"),
    sep="\t"
)

plt.figure(figsize=(10, 10))

gene_freq.head(30).sort_values().plot.barh(color="steelblue")

plt.title("Conserved Neighborhood Genes")
plt.xlabel("Number of Genomes")

plt.tight_layout()

plt.savefig(
    os.path.join(OUTDIR, "gene_neighborhood_frequency.jpg"),
    dpi=600
)

plt.close()


# =============================================================================
# STEP 4: GENOME × GENE MATRIX
# =============================================================================

print("\n[STEP 4] Building presence/absence matrix...")

matrix = pd.crosstab(
    neigh_df["Genome"],
    neigh_df["Gene"]
)

matrix.to_csv(
    os.path.join(OUTDIR, "genome_gene_matrix.tsv"),
    sep="\t"
)


plt.figure(figsize=(14, 12))

sns.heatmap(
    (matrix > 0),
    cmap="viridis",
    cbar=False
)

plt.title("Gene Neighborhood Conservation Map")

plt.tight_layout()

plt.savefig(
    os.path.join(OUTDIR, "gene_neighborhood_heatmap.jpg"),
    dpi=600
)

plt.close()


# =============================================================================
# STEP 5: OPERON CONSTRUCTION
# =============================================================================

print("\n[STEP 5] Inferring operon architectures...")

operons = {}

for genome, g in neigh_df.groupby("Genome"):

    g = g.sort_values("Start")

    operons[genome] = tuple(
        g["Gene"].head(WINDOW_GENES)
    )

operon_counts = Counter(operons.values())

operon_df = pd.DataFrame(
    operon_counts.items(),
    columns=["Operon", "Count"]
).sort_values("Count", ascending=False)

operon_df.to_csv(
    os.path.join(OUTDIR, "operon_architecture.tsv"),
    sep="\t",
    index=False
)


# =============================================================================
# STEP 6: OPERON VISUALIZATION
# =============================================================================

print("\n[STEP 6] Visualizing top operons...")

top_operons = operon_df.head(TOP_OPERONS)

fig, axes = plt.subplots(TOP_OPERONS, 1, figsize=(20, TOP_OPERONS * 2))

if TOP_OPERONS == 1:
    axes = [axes]

for i, (_, row) in enumerate(top_operons.iterrows()):

    genes = row["Operon"]

    features = []
    pos = 0

    for g in genes:

        color = "red" if "rfaC" in g.lower() else "lightblue"

        features.append(
            GraphicFeature(
                start=pos,
                end=pos + 1000,
                strand=+1,
                color=color,
                label=g
            )
        )

        pos += 1200

    record = GraphicRecord(
        sequence_length=pos,
        features=features
    )

    record.plot(ax=axes[i])

    axes[i].set_title(f"Operon {i+1}")

plt.tight_layout()

plt.savefig(
    os.path.join(OUTDIR, "top_operons_diagram.svg"),
    dpi=600
)

plt.close()

print("\n[COMPLETE] Synteny + operon architecture analysis finished successfully.")
