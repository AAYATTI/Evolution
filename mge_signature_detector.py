#!/usr/bin/env python3

# =============================================================================
# HEPTOSYLTRANSFERASE I VARIANTS – MGE ASSOCIATION PIPELINE
# =============================================================================
#
# Author  : Aayatti Mallick Gupta
#
# Description
# -----------------------------------------------------------------------------
# This pipeline identifies and analyzes mobile genetic element (MGE)
# associations surrounding heptosyltransferase I variants, including:
#
#   • OpsX-like systems
#   • Canonical HepI homologs (rfaC / waaC)
#
# The workflow integrates genomic context analysis, distance-based inference,
# and neighborhood scanning to identify MGE enrichment patterns.
#
# -----------------------------------------------------------------------------
# Key Features
# -----------------------------------------------------------------------------
# 1. Detection of MGEs overlapping target loci
# 2. Genomic distance estimation to nearest MGE
# 3. Identification of genomic island proximity
# 4. Gene-neighborhood MGE signature scanning
# 5. Binary presence/absence matrix construction
# 6. Clustered heatmap visualization (publication quality)
#
# -----------------------------------------------------------------------------
# Input
# -----------------------------------------------------------------------------
# gene_neighborhood.tsv
#
# Required columns:
# Genome | Species | Class | Product | Start | End
#
# -----------------------------------------------------------------------------
# Outputs
# -----------------------------------------------------------------------------
# • heptosyltransferase_mge_summary.tsv
# • overlap_heatmap.tsv
# • neighborhood_heatmap.tsv
# • overlap_clustermap.png
# • neighborhood_clustermap.png
#
# =============================================================================


from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# =============================================================================
# CONFIGURATION
# =============================================================================

INPUT_FILE = "gene_neighborhood.tsv"

# Heptosyltransferase I variants
TARGET_CLASSES = ["opsx", "hepi", "rfac", "waac"]

# Genomic island threshold (kb → converted to bp)
GENOMIC_ISLAND_THRESHOLD_KB = 3

# Neighborhood window (gene-level)
UPSTREAM_GENES = 3
DOWNSTREAM_GENES = 3

# Plot styling
HEATMAP_CMAP = "Reds"
FIGSIZE_WIDTH = 12
FONT_SCALE = 0.8


# =============================================================================
# MGE SIGNATURE LIBRARY
# =============================================================================

MGE_KEYWORDS = [
    "transposase", "transposon", "insertion sequence",
    "is1", "is3", "is4", "is5", "is6", "is21", "is30",
    "is66", "is110", "is1595", "is200", "is605",
    "tnpa", "tnpb", "tnpc", "tnpd",

    "integrase", "recombinase", "resolvase",
    "xerc", "xerd", "reca",

    "phage", "capsid", "tail", "portal", "terminase",
    "virion morphogenesis", "baseplate",
    "tape measure", "head protein", "holin", "lysin",

    "type iv secretion", "virb", "virb4", "virb8",
    "virb9", "virb10", "virb11",
    "relaxase", "dna transfer", "plasmid replication",

    "anti-phage"
]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def format_species(name: str) -> str:
    """Convert genome-style names into readable species format."""
    parts = name.split("_")
    if len(parts) >= 3:
        return f"{parts[0].capitalize()} {parts[2].lower()}"
    return name


def compute_genomic_distance(a_start, a_end, b_start, b_end):
    """
    Compute distance between genomic features.

    Returns:
        0 if overlapping, otherwise bp distance.
    """
    if a_end < b_start:
        return b_start - a_end
    elif b_end < a_start:
        return a_start - b_end
    return 0


def build_binary_matrix(df, column):
    """
    Convert semicolon-separated feature column into binary matrix.
    """

    features = sorted({
        item
        for row in df[column]
        for item in str(row).split("; ")
        if item
    })

    matrix = pd.DataFrame(
        0,
        index=df["Species"],
        columns=features
    )

    for _, row in df.iterrows():
        for feature in str(row[column]).split("; "):
            if feature:
                matrix.loc[row["Species"], feature] = 1

    return matrix


def plot_clustermap(matrix, title, output_file):
    """Generate publication-quality clustered heatmap."""

    if matrix.empty:
        print(f"[WARNING] Empty matrix skipped: {output_file}")
        return

    sns.set(font_scale=FONT_SCALE)

    sns.clustermap(
        matrix,
        cmap=HEATMAP_CMAP,
        linewidths=0.5,
        linecolor="gray",
        figsize=(FIGSIZE_WIDTH, max(4, len(matrix) * 0.35)),
        row_cluster=True,
        col_cluster=True
    )

    plt.title(title, y=1.05)

    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[SAVED] {output_file}")


# =============================================================================
# LOAD DATA
# =============================================================================

print("\n[INFO] Loading dataset...")

df = pd.read_csv(INPUT_FILE, sep="\t")

df["Product_lc"] = df["Product"].fillna("").str.lower()
df["Class_lc"] = df["Class"].fillna("").str.lower()

df["is_MGE"] = df["Product_lc"].apply(
    lambda x: any(k in x for k in MGE_KEYWORDS)
)

print(f"[INFO] Total genes loaded: {len(df)}")


# =============================================================================
# IDENTIFY TARGET GENES
# =============================================================================

target_df = df[df["Class_lc"].isin(TARGET_CLASSES)].copy()

print(f"[INFO] Target loci detected: {len(target_df)}")


# =============================================================================
# OVERLAP + DISTANCE ANALYSIS
# =============================================================================

print("\n[INFO] Running overlap analysis...")

summary = []
mge_hits = set()

for _, gene in target_df.iterrows():

    genome = gene["Genome"]
    subset = df[df["Genome"] == genome]
    mges = subset[subset["is_MGE"]]

    overlaps = []
    min_dist = None

    for _, mge in mges.iterrows():

        dist = compute_genomic_distance(
            gene["Start"], gene["End"],
            mge["Start"], mge["End"]
        )

        if dist == 0:
            overlaps.append(mge["Product"])
            mge_hits.add(mge["Product"])

        if min_dist is None or dist < min_dist:
            min_dist = dist

    island = (
        "Yes"
        if min_dist is not None and
        min_dist <= GENOMIC_ISLAND_THRESHOLD_KB * 1000
        else "No"
    )

    summary.append({
        "Species": format_species(gene["Species"]),
        "Genome": genome,
        "Gene": gene["Class"],
        "Overlapping_MGEs": "; ".join(overlaps),
        "Closest_MGE_bp": min_dist if min_dist is not None else "NA",
        "Genomic_Island": island
    })

summary_df = pd.DataFrame(summary)

summary_df.to_csv(
    "heptosyltransferase_mge_summary.tsv",
    sep="\t",
    index=False
)

print("[SAVED] heptosyltransferase_mge_summary.tsv")


# =============================================================================
# NEIGHBORHOOD ANALYSIS
# =============================================================================

print("\n[INFO] Running neighborhood scan...")

neigh = []

for idx, gene in target_df.iterrows():

    genome_df = df[df["Genome"] == gene["Genome"]].reset_index()

    pos = genome_df[genome_df["index"] == idx].index[0]

    window = genome_df.iloc[
        max(0, pos - UPSTREAM_GENES):
        pos + DOWNSTREAM_GENES + 1
    ]

    detected = set()

    for prod in window["Product_lc"]:
        for k in MGE_KEYWORDS:
            if k in prod:
                detected.add(k)

    if detected:
        neigh.append({
            "Species": format_species(gene["Species"]),
            "Genome": gene["Genome"],
            "Gene": gene["Class"],
            "MGEs": "; ".join(sorted(detected))
        })

neigh_df = pd.DataFrame(neigh)


if not neigh_df.empty:

    neigh_df.to_csv(
        "heptosyltransferase_neighborhood.tsv",
        sep="\t",
        index=False
    )

    print("[SAVED] heptosyltransferase_neighborhood.tsv")


# =============================================================================
# HEATMAPS
# =============================================================================

print("\n[INFO] Generating heatmaps...")

if not summary_df.empty:

    heat1 = build_binary_matrix(summary_df, "Overlapping_MGEs")
    heat1.to_csv("overlap_heatmap.tsv", sep="\t")

    plot_clustermap(
        heat1,
        f"GOI–MGE Overlaps (≤{GENOMIC_ISLAND_THRESHOLD_KB} kb)",
        "overlap_clustermap.png"
    )


if not neigh_df.empty:

    heat2 = build_binary_matrix(neigh_df, "MGEs")
    heat2.to_csv("neighborhood_heatmap.tsv", sep="\t")

    plot_clustermap(
        heat2,
        "MGE Signatures in Gene Neighborhood",
        "neighborhood_clustermap.png"
    )


# =============================================================================
# COMPLETION
# =============================================================================

print("\n[INFO] Analysis completed successfully.")
