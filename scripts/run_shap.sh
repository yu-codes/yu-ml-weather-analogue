#!/bin/bash

# A list of Python scripts located in notebooks/experiments to execute
scripts=("notebooks/experiments/shap_total_weighted.py" "notebooks/experiments/shap_total_unweighted.py")

# Activate your virtual environment
echo "-Activating environment..."
source activate ../conda_master_v0

# Initialize a variable to track failed scripts
failed_scripts=()

# Loop through each script and train the model
for script in "${scripts[@]}"; do
    echo "-Starting model training for $script..."
    
    # Run the training script
    python "$script"

    # Check if the script ran successfully
    if [ $? -ne 0 ]; then
        echo "--Training failed for $script. Moving to the next script..."
        failed_scripts+=("$script")
        continue  # Skip to the next script
    fi
    
    echo "-Training completed for $script."
done

# Check if there were any failed scripts
if [ ${#failed_scripts[@]} -ne 0 ]; then
    echo "The following scripts failed:"
    for failed_script in "${failed_scripts[@]}"; do
        echo "-- $failed_script"
    done
else
    echo "All models have been trained successfully!"
fi
