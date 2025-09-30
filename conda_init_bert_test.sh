#!/bin/bash
eval "$(/work/MarkusLundsfrydJensen#1865/miniconda3/bin/conda shell.bash hook)"
conda init
export MY_ENV=blame_bert
echo "conda activate $MY_ENV" >> ~/.bashrc


BASE_DIR="/work/MarkusLundsfrydJensen#1865"
 
# --- GitHub setup ---

cd "$BASE_DIR"

# Install GCM if not present
# now delete any existing gcm files before installing Git Credential Manager. 
if ls *amd* 1> /dev/null 2>&1; then
    echo "GCM files found...deleting before installing new GCM"
    rm *amd*
else
    echo "No files with 'string' found"
fi

#Now git setup.
wget https://github.com/GitCredentialManager/git-credential-manager/releases/download/v2.0.785/gcm-linux_amd64.2.0.785.deb
sudo dpkg -i gcm-linux_amd64.2.0.785.deb

# now configure credential manager. 
git-credential-manager-core configure

sudo apt-get update && sudo apt-get install -y git-lfs
git lfs install
cd /work/MarkusLundsfrydJensen#1865/Bachelor_project
git lfs pull