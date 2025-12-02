# Prerequisites

1. Python 3.7+
2. pip
3. Install required libraries: pip install tensorflow numpy scipy matplotlib

# Usage

1. Prepare your data: Ensure train-1.txt exists in the project folder.
2. Run the script: python lightgcn_v4.py
3. It iterates through the parameter combinations defined in SEARCH_SPACE.

# Output

1. The script identifies the configuration with the highest validation NDCG.
2. Results are saved to submission.txt.
3. A dual-axis plot showing the Loss curve and NDCG accuracy for the best performing model.