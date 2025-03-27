
# Iris Dataset Neural Network Inference with MPI

This project demonstrates how to train a simple two-layer fully connected neural network using the Iris dataset in Python and then perform parallelized inference using an MPI-based C program. The project includes:

- **Python Training Script**: Trains a neural network on the Iris dataset, saves the learned weights (transposed as needed) and test data (`iris_test.csv`).
- **C MPI Inference Program**: Performs inference for a single input sample using parallel row-wise matrix multiplication, including timing measurements.
- **Shell Script**: Extracts the first ten test instances from `iris_test.csv` and calls the MPI inference executable on each instance.
- **README**: This file, providing an overview and usage instructions.

## Files

- **train_model.py**  
  Trains a two-layer fully connected neural network using PyTorch.  
  It saves:
  - The weight file `weights.txt` in a simple text format.
  - The test data `iris_test.csv` (with header and sample rows).

- **parallel.c**  
  The MPI-based C program that performs inference on a single input sample.  
  It reads the weight file, broadcasts weights and a single input sample (passed via command line), computes the forward pass in parallel, and reports the predicted class along with inference time.

- **run_inference.sh**  
  A shell script that reads the first ten instances from `iris_test.csv` (ignoring the header), extracts the four feature values for each instance, and runs the MPI inference executable (`parallel`) for each instance.

- **README.md**  
  This file, which provides an overview of the project and instructions for building and running the code.

## Requirements

- **Python** (3.x) with the following libraries:
  - `torch`
  - `numpy`
  - `scikit-learn`
  
  Install with:
  ```bash
  pip install torch numpy scikit-learn
  ```

- **MPI Library and Compiler** (e.g., OpenMPI and `mpicc`)

## Usage

### 1. Training and Generating Weights
Run the Python training script to train the model and generate the necessary files:
```bash
python3 train.py
```
This will generate:
- `weights.txt` (saved weights for the network)
- `iris_test.csv` (test data)

### 2. Compiling the MPI Inference Program
Compile the C MPI program using `mpicc`:
```bash
mpicc -o parallel parallel.c -lm
```

### 3. Running Inference on Test Instances
Use the provided shell script to run inference on the first ten test instances:
```bash
chmod +x run.sh
./run.sh
```
The script extracts the feature values from each instance in `iris_test.csv` and runs the MPI inference program (`parallel`) using 4 MPI processes. The output for each test instance, including the predicted label and inference time, will be displayed.

## Notes

- **Weight File Format**:  
  The weights are saved in a simple text format that includes:
  - The first layer weights (transposed to a 4×10 matrix)
  - The first layer biases (length 10)
  - The second layer weights (transposed to a 10×3 matrix)
  - The second layer biases (length 3)

- **MPI Inference**:  
  The MPI program performs the matrix multiplication for a single sample by dividing the computation among available processes, using `MPI_Reduce` to combine partial results. Timing is measured using `MPI_Wtime()`.

- **Customization**:  
  You can modify the number of processes in the shell script or change input values as needed.

## Troubleshooting

- **Segmentation Faults**:  
  Ensure that the weight matrices are correctly transposed when saving from Python to match the expected dimensions in the C code.  
- **MPI Errors**:  
  Confirm that your MPI installation is working properly and that you are using the correct command (e.g., `mpirun -np 4 ./parallel`).

## License

This project is provided as-is, for educational purposes.
