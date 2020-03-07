#!/bin/bash

# A self-play script - plays 25 games, stores the training data on a tarball, and repeat for 50000 times

suffix=$1
gpunum=$2

for((i=0;i<50000;i=i+1)) ; do
    resign_rate=$((RANDOM % 10))
    timestamp=$(date +%y%m%d_%H%M%S)_${suffix}
    latest_weight=$(ls -1c training/tf/*.txt | head -1)
    leelaz_cmd="./src/leelaz --acceleration-endgame --gtp -q --precision half -n -d -r $resign_rate -t 5 --batchsize 5 -v 200 --noponder --gpu $gpunum --randomtemp 1.0"
    
    echo $leelaz_cmd
    echo $latest_weight
    echo $timestamp
    
    if(($RANDOM % 2 == 0)) ; then
        echo -e komi 0.5 \\nautotrain training/tf/traindata_${timestamp} 25 \\nquit | ${leelaz_cmd} -w $latest_weight -m 20
    else
        handicap=$((RANDOM % 8 + 2))
        echo -e komi 0.5 \\nfixed_handicap $handicap \\nautotrain training/tf/traindata_${timestamp} 25 \\nquit | ${leelaz_cmd} -w $latest_weight -m $((handicap * 2 + 20))
    fi
done
