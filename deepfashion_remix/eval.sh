#!/bin/bash

disc=$1
arch=$2
savefile="results/$arch$disc.txt"
errfile="results/$arch$disc.err"
batch_size=100

echo $1 $2

python3 validate.py --scales -2 --occlusion -2 --viewpoint -2 --zoom -1 --negate 1 1 1 1 -b $batch_size --arch $arch --disc $disc > $savefile 2> $errfile
python3 validate.py --scales 2 --occlusion 2 --viewpoint 2 --zoom 1 --negate 1 1 1 1 -b $batch_size --concat_data --arch $arch --disc $disc >> $savefile 2>> $errfile
python3 validate.py --scales 2 --occlusion 2 --viewpoint 2 --zoom 1 --negate 0 0 0 0 -b $batch_size --arch $arch --disc $disc >> $savefile 2>> $errfile


python3 validate.py --scales 1 --occlusion 2 --viewpoint 2 --zoom 1 --negate 0 0 0 0 -b $batch_size --arch $arch --disc $disc >> $savefile 2>> $errfile
python3 validate.py --scales 3 --occlusion 2 --viewpoint 2 --zoom 1 --negate 0 0 0 0 -b $batch_size --arch $arch --disc $disc >> $savefile 2>> $errfile
python3 validate.py --scales 2 --occlusion 1 --viewpoint 2 --zoom 1 --negate 0 0 0 0 -b $batch_size --arch $arch --disc $disc >> $savefile 2>> $errfile
python3 validate.py --scales 2 --occlusion 3 --viewpoint 2 --zoom 1 --negate 0 0 0 0 -b $batch_size --arch $arch --disc $disc >> $savefile 2>> $errfile
python3 validate.py --scales 2 --occlusion 2 --viewpoint 1 --zoom 1 --negate 0 0 0 0 -b $batch_size --arch $arch --disc $disc >> $savefile 2>> $errfile
python3 validate.py --scales 2 --occlusion 2 --viewpoint 3 --zoom 1 --negate 0 0 0 0 -b $batch_size --arch $arch --disc $disc >> $savefile 2>> $errfile
python3 validate.py --scales 2 --occlusion 2 --viewpoint 2 --zoom 2 --negate 0 0 0 0 -b $batch_size --arch $arch --disc $disc >> $savefile 2>> $errfile
python3 validate.py --scales 2 --occlusion 2 --viewpoint 2 --zoom 3 --negate 0 0 0 0 -b $batch_size --arch $arch --disc $disc >> $savefile 2>> $errfile

python3 validate.py --scales 2 --occlusion 2 --viewpoint 2 --zoom 1 --negate 0 0 0 0 -b $batch_size  --concat_data --arch $arch --disc $disc >> $savefile 2>> $errfile
python3 validate.py --scales 1 --occlusion 2 --viewpoint 2 --zoom 1 --negate 0 0 0 0 -b $batch_size  --concat_data --arch $arch --disc $disc >> $savefile 2>> $errfile
python3 validate.py --scales 3 --occlusion 2 --viewpoint 2 --zoom 1 --negate 0 0 0 0 -b $batch_size  --concat_data --arch $arch --disc $disc >> $savefile 2>> $errfile
python3 validate.py --scales 2 --occlusion 1 --viewpoint 2 --zoom 1 --negate 0 0 0 0 -b $batch_size  --concat_data --arch $arch --disc $disc >> $savefile 2>> $errfile
python3 validate.py --scales 2 --occlusion 3 --viewpoint 2 --zoom 1 --negate 0 0 0 0 -b $batch_size  --concat_data --arch $arch --disc $disc >> $savefile 2>> $errfile
python3 validate.py --scales 2 --occlusion 2 --viewpoint 1 --zoom 1 --negate 0 0 0 0 -b $batch_size  --concat_data --arch $arch --disc $disc >> $savefile 2>> $errfile
python3 validate.py --scales 2 --occlusion 2 --viewpoint 3 --zoom 1 --negate 0 0 0 0 -b $batch_size  --concat_data --arch $arch --disc $disc >> $savefile 2>> $errfile
python3 validate.py --scales 2 --occlusion 2 --viewpoint 2 --zoom 2 --negate 0 0 0 0 -b $batch_size  --concat_data --arch $arch --disc $disc >> $savefile 2>> $errfile
python3 validate.py --scales 2 --occlusion 2 --viewpoint 2 --zoom 3 --negate 0 0 0 0 -b $batch_size  --concat_data --arch $arch --disc $disc >> $savefile 2>> $errfile


