## Step 1: Prepare data
cd text_classification/data
bash download_dataset.sh
cd ..

## Step 2: Create and activate environment
conda create -n classification python==3.9.16 -y
conda activate classification

# Install dependencies (ignore minor errors and continue)
pip install -r requirements.txt

# Additional libraries
pip install --upgrade --index-url https://download.pytorch.org/whl/cu124 "torch==2.6.*"
pip install numpy==1.24.4 ml_swissknife sentence_transformers
pip install transformers[torch]
pip install transformers==4.45.2

cd ..

## Step 3: Fine-tuning adjustments
# bash run_classification.sh  # uncomment this line to run experiments.
