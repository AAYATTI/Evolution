#!/usr/bin/env python3

# =============================================================================
# PHYLOGENETIC BRANCH LENGTH ANALYSIS OF HEPTOSYLTRANSFERASE I VARIANTS
# =============================================================================
#
# Author  : Aayatti Mallick Gupta
#
# Purpose :
# This pipeline performs comparative phylogenetic branch length analysis
# of coexisting heptosyltransferase I variants, including OpsX-like and
# canonical HepI-like (rfaC/waaC) systems.
#
# The workflow includes:
#   1. Extraction of branch lengths from phylogenetic trees
#   2. Classification of internal and terminal branches
#   3. Between-gene statistical comparisons
#   4. Within-gene branch structure comparisons
#   5. Publication-quality visualization
#   6. Combined phylogenetic tree rendering
#
# Input files :
#   HepI_tree.nwk
#   OpsX_tree.nwk
#
# Outputs :
#   1. branch_length_statistics.log
#   2. branch_length_comparison.png
#   3. branch_length_heatmap.png
#   4. hepi_opsx_combined_tree.png
#
# =============================================================================


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from ete3 import (
    Tree,
    TreeStyle,
    NodeStyle,
    faces
)

from scipy.stats import mannwhitneyu


# =============================================================================
# INPUT FILES
# =============================================================================

HEPI_TREE_FILE = "HepI_tree.nwk"
OPSX_TREE_FILE = "OpsX_tree.nwk"

STATISTICS_LOG = "branch_length_statistics.log"


# =============================================================================
# VISUALIZATION SETTINGS
# =============================================================================

sns.set(style="whitegrid")

BOXPLOT_PALETTE = {
    "HepI-like": "#4C72B0",
    "OpsX-like": "#DD8452"
}


# =============================================================================
# FUNCTION: EXTRACT BRANCH LENGTHS
# =============================================================================

def extract_branch_lengths(
    tree_file,
    gene_label
):
    """
    Extract branch lengths from phylogenetic tree.

    Parameters
    ----------
    tree_file : str
        Newick tree file.

    gene_label : str
        Label assigned to the tree.

    Returns
    -------
    pandas.DataFrame
        Branch length dataset.

    ete3.Tree
        Midpoint-rooted tree object.

    dict
        Counts of internal and terminal branches.
    """

    tree = Tree(tree_file, format=1)

    # Midpoint rooting
    tree.set_outgroup(
        tree.get_midpoint_outgroup()
    )

    branch_data = []

    for node in tree.traverse():

        if node.is_root():
            continue

        branch_type = (
            "terminal"
            if node.is_leaf()
            else "internal"
        )

        branch_data.append({

            "branch_length":
                node.dist,

            "branch_type":
                branch_type,

            "gene_family":
                gene_label
        })

    df = pd.DataFrame(branch_data)

    branch_counts = (
        df["branch_type"]
        .value_counts()
        .to_dict()
    )

    return df, tree, branch_counts


# =============================================================================
# LOAD TREES
# =============================================================================

print("\n[INFO] Loading phylogenetic trees...")

hepi_df, hepi_tree, hepi_counts = extract_branch_lengths(
    HEPI_TREE_FILE,
    "HepI-like"
)

opsx_df, opsx_tree, opsx_counts = extract_branch_lengths(
    OPSX_TREE_FILE,
    "OpsX-like"
)

combined_df = pd.concat(
    [hepi_df, opsx_df],
    ignore_index=True
)

print("[INFO] Trees loaded successfully.")


# =============================================================================
# FUNCTION: BETWEEN-GENE COMPARISON
# =============================================================================

def compare_between_families(
    dataframe,
    branch_type
):
    """
    Compare branch lengths between HepI-like and OpsX-like systems.
    """

    hepi = dataframe[
        (dataframe["gene_family"] == "HepI-like") &
        (dataframe["branch_type"] == branch_type)
    ]["branch_length"]

    opsx = dataframe[
        (dataframe["gene_family"] == "OpsX-like") &
        (dataframe["branch_type"] == branch_type)
    ]["branch_length"]

    statistic, pvalue = mannwhitneyu(
        hepi,
        opsx,
        alternative="two-sided"
    )

    return (
        statistic,
        pvalue,
        hepi.mean(),
        opsx.mean()
    )


# =============================================================================
# FUNCTION: WITHIN-GENE COMPARISON
# =============================================================================

def compare_branch_structure(
    dataframe,
    gene_family
):
    """
    Compare internal vs terminal branch lengths
    within a gene family.
    """

    internal = dataframe[
        (dataframe["gene_family"] == gene_family) &
        (dataframe["branch_type"] == "internal")
    ]["branch_length"]

    terminal = dataframe[
        (dataframe["gene_family"] == gene_family) &
        (dataframe["branch_type"] == "terminal")
    ]["branch_length"]

    statistic, pvalue = mannwhitneyu(
        internal,
        terminal,
        alternative="two-sided"
    )

    return (
        statistic,
        pvalue,
        internal.mean(),
        terminal.mean()
    )


# =============================================================================
# STATISTICAL ANALYSIS
# =============================================================================

print("\n[INFO] Performing statistical analysis...")

# Between-family comparisons
(
    internal_stat,
    internal_p,
    internal_hepi_mean,
    internal_opsx_mean
) = compare_between_families(
    combined_df,
    "internal"
)

(
    terminal_stat,
    terminal_p,
    terminal_hepi_mean,
    terminal_opsx_mean
) = compare_between_families(
    combined_df,
    "terminal"
)

# Within-family comparisons
(
    hepi_stat,
    hepi_p,
    hepi_internal_mean,
    hepi_terminal_mean
) = compare_branch_structure(
    combined_df,
    "HepI-like"
)

(
    opsx_stat,
    opsx_p,
    opsx_internal_mean,
    opsx_terminal_mean
) = compare_branch_structure(
    combined_df,
    "OpsX-like"
)


# =============================================================================
# SAVE STATISTICS
# =============================================================================

print("[INFO] Writing statistical summary...")

with open(STATISTICS_LOG, "w") as log:

    log.write(
        "====================================================\n"
    )

    log.write(
        "PHYLOGENETIC BRANCH LENGTH ANALYSIS\n"
    )

    log.write(
        "Heptosyltransferase I Variants\n"
    )

    log.write(
        "====================================================\n\n"
    )

    # Branch counts
    log.write(
        "=== Branch Counts ===\n"
    )

    log.write(
        f"HepI-like : {hepi_counts}\n"
    )

    log.write(
        f"OpsX-like : {opsx_counts}\n\n"
    )

    # Between-family comparisons
    log.write(
        "=== Between-Family Comparisons ===\n"
    )

    log.write(
        (
            "Internal branches : "
            f"HepI-like mean = {internal_hepi_mean:.4f}, "
            f"OpsX-like mean = {internal_opsx_mean:.4f}, "
            f"Mann-Whitney p = {internal_p:.4g}\n"
        )
    )

    log.write(
        (
            "Terminal branches : "
            f"HepI-like mean = {terminal_hepi_mean:.4f}, "
            f"OpsX-like mean = {terminal_opsx_mean:.4f}, "
            f"Mann-Whitney p = {terminal_p:.4g}\n\n"
        )
    )

    # Within-family comparisons
    log.write(
        "=== Within-Family Comparisons ===\n"
    )

    log.write(
        (
            "HepI-like : "
            f"Internal mean = {hepi_internal_mean:.4f}, "
            f"Terminal mean = {hepi_terminal_mean:.4f}, "
            f"Mann-Whitney p = {hepi_p:.4g}\n"
        )
    )

    log.write(
        (
            "OpsX-like : "
            f"Internal mean = {opsx_internal_mean:.4f}, "
            f"Terminal mean = {opsx_terminal_mean:.4f}, "
            f"Mann-Whitney p = {opsx_p:.4g}\n"
        )
    )

print(f"[SAVED] {STATISTICS_LOG}")


# =============================================================================
# PLOT:
# BRANCH LENGTH DISTRIBUTION
# =============================================================================

print("\n[INFO] Generating branch length plots...")

plt.figure(figsize=(8, 6))

sns.boxplot(
    x="branch_type",
    y="branch_length",
    hue="gene_family",
    data=combined_df,
    palette=BOXPLOT_PALETTE
)

sns.swarmplot(
    x="branch_type",
    y="branch_length",
    hue="gene_family",
    data=combined_df,
    dodge=True,
    color=".25",
    alpha=0.6
)

plt.title(
    "Phylogenetic Branch Length Comparison\n"
    "HepI-like vs OpsX-like Systems"
)

plt.xlabel("Branch type")
plt.ylabel("Branch length")

plt.legend(
    title="Gene family",
    bbox_to_anchor=(1.05, 1),
    loc="upper left"
)

plt.tight_layout()

plt.savefig(
    "branch_length_comparison.png",
    dpi=300
)

plt.close()

print("[SAVED] branch_length_comparison.png")


# =============================================================================
# COMBINED TREE VISUALIZATION
# =============================================================================

print("\n[INFO] Rendering combined phylogenetic tree...")

combined_tree = hepi_tree.copy()

# Add OpsX-like leaves
for leaf in opsx_tree.get_leaves():
    combined_tree.add_child(
        leaf.detach()
    )

hepi_leaf_names = {
    leaf.name
    for leaf in hepi_tree.get_leaves()
}

opsx_leaf_names = {
    leaf.name
    for leaf in opsx_tree.get_leaves()
}

for leaf in combined_tree:

    style = NodeStyle()

    style["size"] = 5

    if leaf.name in hepi_leaf_names:
        style["fgcolor"] = "blue"

    elif leaf.name in opsx_leaf_names:
        style["fgcolor"] = "red"

    else:
        style["fgcolor"] = "black"

    leaf.set_style(style)

tree_style = TreeStyle()

tree_style.show_leaf_name = True
tree_style.scale = 50

tree_style.title.add_face(
    faces.TextFace(
        (
            "Combined Phylogeny:\n"
            "HepI-like (blue) vs OpsX-like (red)"
        ),
        fsize=12
    ),
    column=0
)

combined_tree.render(
    "hepi_opsx_combined_tree.png",
    w=900,
    tree_style=tree_style
)

print("[SAVED] hepi_opsx_combined_tree.png")


# =============================================================================
# HEATMAP:
# MEAN BRANCH LENGTHS
# =============================================================================

print("\n[INFO] Generating branch length heatmap...")

heatmap_df = combined_df.pivot_table(
    index="gene_family",
    columns="branch_type",
    values="branch_length",
    aggfunc="mean"
)

plt.figure(figsize=(4, 3))

sns.heatmap(
    heatmap_df,
    annot=True,
    cmap="viridis"
)

plt.title(
    "Mean Phylogenetic Branch Lengths"
)

plt.tight_layout()

plt.savefig(
    "branch_length_heatmap.png",
    dpi=300
)

plt.close()

print("[SAVED] branch_length_heatmap.png")


# =============================================================================
# COMPLETED
# =============================================================================

print("\n[INFO] Analysis completed successfully.")
