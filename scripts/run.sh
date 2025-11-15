rm -r "results"
mkdir "results"
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh

conda activate DL
if [ $? -ne 0 ]; then
    echo "Failed to activate Conda environment. Exiting."
    exit 1
fi
models=("CNN-1D" "SSL")

user_configs=("0,1" "0,1,2,3,4,5")

# Function to run models for a specific user configuration
run_models_for_users() {
    local users=$1
    echo "Running models for users: $users"

    for model in "${models[@]}"; do
        echo "Running model: $model"

        python run.py --model "$model" --users "$users"

        mv results/result.json "results/result_${model}_users_${users// /_}.json"

        echo "Completed running $model for users $users. Results saved in result_${model}_users_${users// /_}.json"
        echo "----------------------------------------"
    done
}

for config in "${user_configs[@]}"; do
    run_models_for_users "$config"
    echo "========================================="
done
