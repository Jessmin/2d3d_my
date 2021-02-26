conda create -n my_env python=3.8 -y
conda activate my_env
conda install pytorch torchvision torchaudio cudatoolkit=11.0 -c pytorch -c=conda-forge -y
pip install opencv-python opencv-contrib-python matplotlib packaging
conda install numba -y