import re
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

def extract_dn_ds_from_file(dn_ds_file_path):
    """
    Extract average dN/dS values from the input file.
    """
    dn_ds_values = []
    with open(dn_ds_file_path, 'r') as file:
        lines = file.readlines()
        for line_number, line in enumerate(lines):
            match = re.search(r'Average dN/dS for .*:\s*([\d.]+)', line)
            if match:
                try:
                    value = float(match.group(1))
                    dn_ds_values.append(value)
                except ValueError:
                    continue
    return dn_ds_values

def extract_cai(cai_file_path):
    """
    Extract CAI values from a log file for individual sequences.
    """
    cai_values = []
    with open(cai_file_path, 'r') as file:
        lines = file.readlines()
        for line_number, line in enumerate(lines):
            if line.startswith("Gene"):
                continue
            try:
                value = float(line.strip().split()[-1])
                cai_values.append(value)
            except (ValueError, IndexError):
                continue
    return cai_values

# File paths
dn_ds_files = {
    "Higher": "dn_ds-higher.txt",
    "Intermediate": "dn_ds-intermediate.txt",
    "Lower": "dn_ds-lower.txt"
}

cai_files = {
    "Higher": "cai-higher.log",
    "Intermediate": "cai-intermediate.log",
    "Lower": "cai-lower.log"
}

# Colors for scatter plot
colors = {
    "Higher": "blue",
    "Intermediate": "green",
    "Lower": "maroon"
}

# Combine all data
all_data = []

for category in dn_ds_files:
    dn_ds_values = extract_dn_ds_from_file(dn_ds_files[category])
    cai_values = extract_cai(cai_files[category])

    if len(dn_ds_values) != len(cai_values):
        print(f"[Error] The number of dN/dS values does not match CAI values for {category}.")
        continue

    all_data.append(pd.DataFrame({
        "dN/dS": dn_ds_values,
        "CAI": cai_values,
        "Category": category
    }))

# Combine all categories into a single DataFrame
final_data = pd.concat(all_data)

# Save correlation results to a file
with open("correlation_results.txt", "w") as result_file:
    for category in dn_ds_files:
        category_data = final_data[final_data['Category'] == category]
        if not category_data.empty:
            pearson_corr, _ = stats.pearsonr(category_data['dN/dS'], category_data['CAI'])
            spearman_corr, _ = stats.spearmanr(category_data['dN/dS'], category_data['CAI'])
            result_file.write(f"[{category}] Pearson correlation: {pearson_corr}\n")
            result_file.write(f"[{category}] Spearman correlation: {spearman_corr}\n")
            print(f"[{category}] Pearson correlation: {pearson_corr}")
            print(f"[{category}] Spearman correlation: {spearman_corr}")

# Plotting
plt.figure(figsize=(10, 6))
for category in dn_ds_files:
    category_data = final_data[final_data['Category'] == category]
    plt.scatter(category_data['dN/dS'], category_data['CAI'], label=category, color=colors[category])

plt.xlabel("dN/dS")
plt.ylabel("CAI")
plt.title("Correlation between dN/dS and CAI")
plt.legend()
plt.savefig("multi_correlation_plot.png")
plt.show()

