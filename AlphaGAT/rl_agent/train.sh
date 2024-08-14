#!/bin/bash

#for drop in {0.0,0.2}
#do
drop=0.2
for lr in {5e-4,1e-5}
do
for G in {1,5,10}
do
for adv in {0.95,0.99}
do
for entropy in {0.01,0.005,0.001}
do
#alphas=256
    python train_agent.py  --save_dir "./dp_${drop}/lr_${lr}/G_${G}/adv_${adv}/entropy_${entropy}"\
		    --batch_size 512\
		    --horizon_len 2048\
		    --break_step 400000\
		    --learning_rate  "${lr}"\
		    --drop "${drop}"\
		    --G "${G}"\
		    --repeat_times 8.0\
		    --adv "${adv}"\
		    --entropy "${entropy}"> ./LOGS/dp_${drop}lr_${lr}G_${G}adv_${adv}entropy_${entropy}.out
done
done
done
done
