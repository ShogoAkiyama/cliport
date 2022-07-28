#!/bin/bash

DATA_DIR=$1
DISP=False

echo "Generating dataset... Folder: $DATA_DIR"

# You can parallelize these depending on how much resources you have

#############################
## Language-Conditioned Tasks

LANG_TASKS='packing-boxes-pairs-seen-colors'

for task in $LANG_TASKS
    do
        python cliport/demos.py n=1000 task=$task mode=train data_dir=$DATA_DIR disp=$DISP &
        python cliport/demos.py n=100  task=$task mode=val   data_dir=$DATA_DIR disp=$DISP &
        python cliport/demos.py n=100  task=$task mode=test  data_dir=$DATA_DIR disp=$DISP
    done
echo "Finished Language Tasks."


# #########################
# ## Demo-Conditioned Tasks

# DEMO_TASKS='align-box-corner assembling-kits block-insertion manipulating-rope packing-boxes palletizing-boxes place-red-in-green stack-block-pyramid sweeping-piles towers-of-hanoi'

# for task in $DEMO_TASKS
#     do
#         python cliport/demos.py n=1000 task=$task mode=train data_dir=$DATA_DIR disp=$DISP &
#         python cliport/demos.py n=100  task=$task mode=val   data_dir=$DATA_DIR disp=$DISP &
#         python cliport/demos.py n=100  task=$task mode=test  data_dir=$DATA_DIR disp=$DISP
#     done
# echo "Finished Demo Tasks."
