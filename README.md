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
unzip video_train.zip -d store/video_generation/CLEVREV/video_train;
# unzipping the validation dataset
unzip video_validation.zip -d store/video_generation/CLEVREV/video_validation;
# unzipping the test dataset
unzip video_test.zip -d store/video_generation/CLEVREV/video_test;
# removing zipped archives
rm *.zip;
```