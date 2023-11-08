#!/bin/bash

# Check if exactly three arguments are provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <M> <N> <K>"
    exit 1
fi

# Extract M, N, and K from the command-line arguments
M="$1"
N="$2"
K="$3"

# Define an array of possible values (M, N, K)
values=("$M" "$N" "$K")

permutations=(
    "$M $N $K"
    "$M $K $N"
    "$N $M $K"
    "$N $K $M"
    "$K $M $N"
    "$K $N $M"
)

# Generate all possible permutations of the first 3 arguments
for args in "${permutations[@]}"; do
    # Run ./a.out with the first 3 arguments and 3 additional arguments
    for additional_arg1 in "${values[@]}"; do
        for additional_arg2 in "${values[@]}"; do
            for additional_arg3 in "${values[@]}"; do
                echo "TRYING:" $args $additional_arg1 $additional_arg2 $additional_arg3
                ./a.out $args $additional_arg1 $additional_arg2 $additional_arg3
            done
        done
    done
done
