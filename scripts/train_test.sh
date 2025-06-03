#!/bin/bash

# A list of Python scripts located in notebooks/experiments to execute
scripts=("notebooks/experiments/exp_1.py" "notebooks/experiments/exp_2.py" "notebooks/experiments/exp_3.py")

# Activate your virtual environment
echo "-Activating environment..."
source activate ../conda_master_v0

# Loop through each script and train the model
for script in "${scripts[@]}"; do
    echo "-Starting model training for $script..."

    # Run the training script
    python "$script"

    # Check if the script ran successfully
    if [ $? -ne 0 ]; then
        echo "--Training failed for $script."
        exit 1
    fi

    echo "-Training completed for $script."
done

echo "All models have been trained successfully!"
