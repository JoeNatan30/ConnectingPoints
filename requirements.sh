#conda create --name signs python=3.8
#conda activate signs

pip install --upgrade pip
pip install youtube-dl
pip install opencv-python
pip install pandas
pip install pytorch-lightning
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install mediapipe
pip install h5py
conda install -c anaconda cmake
conda install -c conda-forge yacs
pip install gdown