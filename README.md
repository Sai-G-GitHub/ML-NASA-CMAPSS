# Predictive Maintenance Using LSTM for Remaining Useful Life (RUL) Prediction

## Overview

This project implements a predictive maintenance model using a bidirectional Long Short-Term Memory (LSTM) neural network to estimate the Remaining Useful Life (RUL) of engines. The model was trained and tested on large-scale time-series datasets, consisting of over 800,000 lines and 15 million data points, sourced from the [CMAPSS dataset](https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository).

## Project Structure

- **Data Preprocessing**: The data is preprocessed using Pandas, including scaling and sequence padding, to prepare it for time-series analysis.
- **Model Architecture**: A custom bidirectional LSTM model is implemented using TensorFlow, incorporating masking, dropout, and batch normalization layers.
- **Training and Optimization**: The model is trained using advanced techniques such as early stopping, learning rate reduction, and model checkpointing to ensure robust performance.
- **Evaluation**: The model’s performance is evaluated using Mean Absolute Error (MAE) on both scaled and original data, ensuring practical relevance.

## Key Features

- **Large Dataset Handling**: Efficiently processes and scales large datasets (800,000+ lines, 15M+ data points) for time-series prediction tasks.
- **Custom LSTM Model**: Implements a bidirectional LSTM with advanced features like masking and dropout to handle variable-length sequences.
- **Optimization Techniques**: Utilizes callbacks such as early stopping, ReduceLROnPlateau, and ModelCheckpoint to optimize model training and prevent overfitting.
- **Sequence Padding**: Handles variable-length sequences by padding to a fixed length, ensuring consistent input size for the LSTM.

## Installation

Clone this repository:

```bash
git clone https://github.com/Sai-G-GitHub/ML-NASA-CMAPSS
```

Install the required Python packages:

- NumPy, Pandas, Scikit-learn, TensorFlow

## Usage

1. **Data Preparation**: Place the CMAPSS dataset files in the appropriate directory (`cmapss`).
2. **Training**: Run the preprocessing and model training scripts to train the LSTM model.
3. **Evaluation**: Evaluate the model’s performance using the test dataset.


## Results

- The model achieved a Mean Absolute Error (MAE) of [0.03] on the test set.
- The trained model files and performance metrics are saved in the `processed` directory.

## Project Details

- **Technologies Used**: Python, TensorFlow, Pandas, NumPy, scikit-learn
- **Data**: CMAPSS dataset (800,000+ lines, 15M+ data points)
- **Model**: Bidirectional LSTM with masking, dropout, and batch normalization

## Future Work

- **Hyperparameter Tuning**: Experiment with different LSTM architectures and hyperparameters.
- **Deployment**: Convert the trained model to a format suitable for deployment in production environments.
- **Visualization**: Implement visualization tools to better understand the model's predictions.

## License

This project is licensed under the MIT License.

## Contact

For any questions or feedback, feel free to reach out at [sagruto@gmail.com].
