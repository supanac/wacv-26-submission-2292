#!/bin/bash -l

#SBATCH --partition=singlenode
#SBATCH --time=08:00:00
#SBATCH --output /home/hpc/b105dc/b105dc10/gesture-tokenizer/asd_logs/asd_on_rpd_video_id_%x_job_id_%j_fritz.out
#SBACTH --export=NONE

unset SLURM_EXPORT_ENV

export LANG='en_US.UTF-8'
export LC_ALL='en_US.UTF-8'


RUN_FOLDER=$(pwd)

module load python
cd /tmp/$SLURM_JOB_ID.fritz
python -m venv venvs/foo
source venvs/foo/bin/activate
python -m pip install gdown scikit-learn --quiet
python -m pip install torch numpy scipy python_speech_features opencv-python facenet-pytorch pandas tqdm --quiet
python -m pip install --upgrade scenedetect[opencv] --quiet
cd $RUN_FOLDER

ffmpeg=/home/atuin/b105dc/data/software/ffmpeg/ffmpeg
ALL_VIDEOS_LIST=/home/atuin/b105dc/data/datasets/russian_propaganda_dataset_openpose/all_russian_propaganda_dataset.txt
ALL_SCENES_FOLDER=/home/atuin/b105dc/data/datasets/russian_propaganda_dataset_openpose/scenes
THREADS=10

CASE_ID=$1
DEVICE=cpu

echo $CASE_ID

VIDEO_PATH=$(grep "$CASE_ID" "$ALL_VIDEOS_LIST")
echo $VIDEO_PATHecho $VIDEO_PATH
SAVE_TO=$DATASET/russian_propaganda_dataset_openpose/test_asd/$CASE_ID
PATH_TO_SCENES=$ALL_SCENES_FOLDER/$CASE_ID-Scenes.csv

mkdir $TMPDIR/{pyavi,pyframes,pywork}
if [ ! -d "$SAVE_TO" ]; then
    mkdir $SAVE_TO
fi

$ffmpeg -y -i $VIDEO_PATH -qscale:v 2 -threads $THREADS -async 1 -r 25 $TMPDIR/temp_video_25fps.avi -loglevel panic
$ffmpeg -y -i $VIDEO_PATH -qscale:a 0 -ac 1 -vn -threads $THREADS -ar 16000 $TMPDIR/temp_audio_16khz.wav -loglevel panic
$ffmpeg -y -i $VIDEO_PATH -qscale:v 2 -threads $THREADS -f image2 $TMPDIR/pyframes/%07d.jpg -loglevel panic

cd /home/atuin/b105dc/data/work/iburenko/talknet_optimised/TalkNet-ASD
python run_talknet.py \
    --pathToScenes $PATH_TO_SCENES \
    --videoName temp_video_25fps \
    --videoFolder $TMPDIR \
    --audioFilePath $TMPDIR/temp_audio_16khz.wav \
    --device $DEVICE 

cp $TMPDIR/pywork/* $SAVE_TO/
if [ -f $TMPDIR/pyavi/video_out.avi ]; then 
    cp $TMPDIR/pyavi/video_out.avi $SAVE_TO/$CASE_ID"_"$DEVICE".avi"
fi