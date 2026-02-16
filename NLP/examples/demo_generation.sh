## Step 1: Prepare data
# The dataset is originally released under CC BY-SA 4.0 (for E2E) and MIT (for DART).
# We provide this temporary link solely for review purposes, with full attribution to the original authors.
# temporary link: https://uconn-my.sharepoint.com/:u:/g/personal/qin_yang_uconn_edu/EVm7PNcQQ3pLozAvCVWTzyQBi4lQEScyQQk7pkkOgV-KNg?e=CGp8Yl

# Please ensure the folder structure:
# table2text/prefix-tuning/data/e2e_data/

## Step 2: Create and activate environment
conda create -n generation python==3.9.16 -y
conda activate generation

# Install dependencies (ignore minor errors and continue)
cd table2text
pip install -r requirements.txt

# # Additional libraries
pip install fire ml_swissknife deepspeed
pip install --upgrade --index-url https://download.pytorch.org/whl/cu124 "torch==2.6.*"
pip install numpy==1.24.4
pip install transformers==4.38.2
pip install transformers[torch]

cd ..

## Step 3: Fine-tuning adjustments
# bash run_generation.sh  # uncomment this line to run experiments.
