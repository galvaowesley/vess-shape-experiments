# 1. Clone torchtrainer
if [ ! -d "torchtrainer" ]; then
    git clone https://github.com/chcomin/torchtrainer.git
fi

# 2. Install torchtrainer
pip install -e torchtrainer

# 3. Clone vessel-shape-dataset
if [ ! -d "vess-shape-dataset" ]; then
    git clone git@github.com:galvaowesley/vess-shape-dataset.git
fi

# 4. Install vessel-shape-dataset package
pip install -e vess-shape-dataset

