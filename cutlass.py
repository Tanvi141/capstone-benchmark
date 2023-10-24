import pandas as pd

file_path = 'out.gemm.csv'
batched_file_path = 'vary_k_batch_out.gemm.csv'

df = pd.read_csv(file_path)
batch_df = pd.read_csv(batched_file_path)

def get_max_tflops(m, n, k): 

    # Filter the DataFrame to rows where m, n, and k match the input values
    filtered_df = df[(df['m'] == m) & (df['n'] == n) & (df['k'] == k)] 

    # Check if any rows match the criteria
    if not filtered_df.empty:
        max_tflops = filtered_df['GFLOPs'].max() / 1000
        return max_tflops
    else:
        return -1  # No matching rows found

def get_max_tflops_batched(m, n, k, b): 

    # Filter the DataFrame to rows where m, n, and k match the input values
    filtered_df = batch_df[(batch_df['m'] == m) & (batch_df['n'] == n) & (batch_df['k'] == k) & (batch_df['batch_count'] == b)] 

    # Check if any rows match the criteria
    if not filtered_df.empty:
        max_tflops = filtered_df['GFLOPs'].max() / 1000
        return max_tflops
    else:
        return None  # No matching rows found

