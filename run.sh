#!/bin/bash
# run_inference.sh
#
# This script reads the first ten instances from iris_test.csv (ignoring the header),
# extracts the first four feature values from each instance, and then calls the MPI
# inference executable "parallel" with those values as arguments.
# The output for each instance is printed to the terminal.

# Check if iris_test.csv exists
if [ ! -f "iris_test.csv" ]; then
    echo "Error: iris_test.csv not found."
    exit 1
fi

# Check if the MPI executable exists and is executable
if [ ! -x "./parallel" ]; then
    echo "Error: MPI executable 'parallel' not found or not executable."
    exit 1
fi

echo "Running inference on the first ten instances from iris_test.csv"
echo "-------------------------------------"

# Read the first ten instances (skip header)
count=0
tail -n +2 iris_test.csv | head -n 10 | while IFS= read -r line; do
    # Expecting CSV format: f1,f2,f3,f4,label
    IFS=',' read -r f1 f2 f3 f4 label <<< "$line"
    count=$((count+1))
    echo "Instance $count: Features: $f1, $f2, $f3, $f4 (Label: $label)"
    
    echo "Running MPI inference for instance $count..."
    mpirun -np 4 ./parallel "$f1" "$f2" "$f3" "$f4"
    echo "-------------------------------------"
done
