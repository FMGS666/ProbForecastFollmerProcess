# ProbForecastFollmerProcess

## Video Generation
```{bash}
# downloading the CLEVREV train dataset
wget http://data.csail.mit.edu/clevrer/videos/train/video_train.zip;
# downloading the CLEVREV validation dataset
wget http://data.csail.mit.edu/clevrer/videos/validation/video_validation.zip;
# downloading the CLEVREV test dataset
wget http://data.csail.mit.edu/clevrer/videos/test/video_test.zip;
# unzipping the train dataset
unzip video_train.zip -d store/video_generation/data/CLEVREV/video_train;
# unzipping the validation dataset
unzip video_validation.zip -d store/video_generation/data/CLEVREV/video_validation;
# unzipping the test dataset
unzip video_test.zip -d store/video_generation/data/CLEVREV/video_test;
# removing zipped archives
rm *.zip;
# downloading the pretrained RIVER VQVAE weights on the CLEVRER dataset
wget https://huggingface.co/cvg-unibe/river_clevrer_128/resolve/main/vqvae.pth -O store/video_generation/weights/CLEVREV/vqvae.pth;
# downloading the pretrained RIVER VQGAN weights on the CLEVREV dataset
wget https://huggingface.co/cvg-unibe/river_clevrer_128/resolve/main/model.pth -O store/video_generation/weights/CLEVREV/model.pth;
```