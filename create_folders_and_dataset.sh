# Use this sh to have WLASL, aec and pucp dataset
# WLASL need more process to completely have the data

mkdir datasets

cd datasets

gdown https://drive.google.com/uc?id=1WHxKijB8t5JLljM59hAqi5KY0U6d7OzA -O aec.zip
unzip aec.zip

gdown https://drive.google.com/uc?id=1z5C6qqTRVWgcGzizJSAVEJyn1DI5lGnt -O pucp.zip
unzip pucp.zip

git clone https://github.com/dxli94/WLASL.git

cd ..

mkdir output
gdown https://drive.google.com/uc?id=1q1flAHyPnGB7MfPEJv0P9mMKCJkrnBnr
