conda create -n my_env python =3.9 -y 
conda activate my_env
conda install pytorch torchvision torchaudio cudatoolkit=11.0 -c pytorch -c=conda-forge -y
pip install opencv-pythonl