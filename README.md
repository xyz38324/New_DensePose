# New_DensePose
# install 
conda install pytorch
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.8/index.html


#install densepose 
pip install git+https://github.com/facebookresearch/detectron2@main#subdirectory=projects/DensePose
#install other depedencies
conda install chardet

#download pretrained weight
 from 