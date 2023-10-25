#!/bin/bash

# Define the list of values for each variable
m_block_values=(32)
n_block_values=(32)
k_block_values=(16)
m_group_values=(8)
stages_values=(3)
warps_values=(8)

# Output file for results
output_file="benchmark_results.txt"

# Iterate over the combinations of values
for m_block in "${m_block_values[@]}"; do
  for n_block in "${n_block_values[@]}"; do
    for k_block in "${k_block_values[@]}"; do
      for m_group in "${m_group_values[@]}"; do
        for stages in "${stages_values[@]}"; do
          for warps in "${warps_values[@]}"; do
            # Call the Python script with the current combination of values
            args="$m_block,$n_block,$k_block,$m_group,$stages,$warps"
            python3 benchmark_configs.py "$args"
          done
        done
      done
    done
  done
done

echo "Script finished. Results are in $output_file"
