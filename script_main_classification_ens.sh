#!/bin/bash


# bash script.sh
# tmux attach -t script_mol_opt
# tmux detach
# pkill python

# bash script_main_molecules_graph_regression_ZINC.sh


############
# GNNs
############

#GatedGCN
#GCN
#GraphSage
#MLP
#GIN
#MoNet
#GAT
#DiffPool


############
# ZINC - 8 RUNS
############

seed0=41
seed1=95
seed2=12
seed3=35
seed4=42
seed5=96
seed6=13
seed7=36
#code=main_molecules_graph_regression.py 
#tmux new -s benchmark_molecules_graph_regression -d
tmux new -s bbb_test_ens -d
tmux send-keys "source activate benchmark_gnn" C-m

#   SWA + dropout

SEQ=$(seq 0 9)

for sid in $SEQ
do  
    code=main_classification.py

    dataset=BBBP
    config=gin_bbbp

    numL=4
    wd=0.
    residual=True
    epochs=200
    bbp_complexity=0.01
    bbp_prior_sigma_1=1.


    tmux send-keys "
    python $code --dataset $dataset --gpu_id 0 --seed $sid --data_seed $seed0 --config 'configs/${config}.json' --L $numL --residual $residual --weight_decay $wd --epochs $epochs --bbp True --bbp_complexity $bbp_complexity --bbp_prior_sigma_1 $bbp_prior_sigma_1 & 
    python $code --dataset $dataset --gpu_id 1 --seed $sid --data_seed $seed1 --config 'configs/${config}.json' --L $numL --residual $residual --weight_decay $wd --epochs $epochs --bbp True --bbp_complexity $bbp_complexity --bbp_prior_sigma_1 $bbp_prior_sigma_1 & 
    python $code --dataset $dataset --gpu_id 2 --seed $sid --data_seed $seed2 --config 'configs/${config}.json' --L $numL --residual $residual --weight_decay $wd --epochs $epochs --bbp True --bbp_complexity $bbp_complexity --bbp_prior_sigma_1 $bbp_prior_sigma_1 & 
    python $code --dataset $dataset --gpu_id 3 --seed $sid --data_seed $seed3 --config 'configs/${config}.json' --L $numL --residual $residual --weight_decay $wd --epochs $epochs --bbp True --bbp_complexity $bbp_complexity --bbp_prior_sigma_1 $bbp_prior_sigma_1 & 
    python $code --dataset $dataset --gpu_id 4 --seed $sid --data_seed $seed4 --config 'configs/${config}.json' --L $numL --residual $residual --weight_decay $wd --epochs $epochs --bbp True --bbp_complexity $bbp_complexity --bbp_prior_sigma_1 $bbp_prior_sigma_1 & 
    python $code --dataset $dataset --gpu_id 5 --seed $sid --data_seed $seed5 --config 'configs/${config}.json' --L $numL --residual $residual --weight_decay $wd --epochs $epochs --bbp True --bbp_complexity $bbp_complexity --bbp_prior_sigma_1 $bbp_prior_sigma_1 & 
    python $code --dataset $dataset --gpu_id 6 --seed $sid --data_seed $seed6 --config 'configs/${config}.json' --L $numL --residual $residual --weight_decay $wd --epochs $epochs --bbp True --bbp_complexity $bbp_complexity --bbp_prior_sigma_1 $bbp_prior_sigma_1 & 
    python $code --dataset $dataset --gpu_id 7 --seed $sid --data_seed $seed7 --config 'configs/${config}.json' --L $numL --residual $residual --weight_decay $wd --epochs $epochs --bbp True --bbp_complexity $bbp_complexity --bbp_prior_sigma_1 $bbp_prior_sigma_1 & 
    wait" C-m

done

SEQ=$(seq 0 9)

for sid in $SEQ
do  

    code=main_classification.py

    dataset=BBBP
    config=gin_bbbp

    numL=4
    wd=0.
    residual=True
    epochs=200
    bbp_complexity=0.01
    bbp_prior_sigma_1=1.

    tmux send-keys "
    python $code --dataset $dataset --gpu_id 0 --seed $sid --data_seed $seed0 --config 'configs/${config}.json' --L $numL --residual $residual --weight_decay $wd --epochs $epochs --bbp True --bbp_complexity $bbp_complexity --bbp_prior_sigma_1 $bbp_prior_sigma_1 & 
    python $code --dataset $dataset --gpu_id 1 --seed $sid --data_seed $seed1 --config 'configs/${config}.json' --L $numL --residual $residual --weight_decay $wd --epochs $epochs --bbp True --bbp_complexity $bbp_complexity --bbp_prior_sigma_1 $bbp_prior_sigma_1 & 
    python $code --dataset $dataset --gpu_id 2 --seed $sid --data_seed $seed2 --config 'configs/${config}.json' --L $numL --residual $residual --weight_decay $wd --epochs $epochs --bbp True --bbp_complexity $bbp_complexity --bbp_prior_sigma_1 $bbp_prior_sigma_1 & 
    python $code --dataset $dataset --gpu_id 3 --seed $sid --data_seed $seed3 --config 'configs/${config}.json' --L $numL --residual $residual --weight_decay $wd --epochs $epochs --bbp True --bbp_complexity $bbp_complexity --bbp_prior_sigma_1 $bbp_prior_sigma_1 & 
    python $code --dataset $dataset --gpu_id 4 --seed $sid --data_seed $seed4 --config 'configs/${config}.json' --L $numL --residual $residual --weight_decay $wd --epochs $epochs --bbp True --bbp_complexity $bbp_complexity --bbp_prior_sigma_1 $bbp_prior_sigma_1 & 
    python $code --dataset $dataset --gpu_id 5 --seed $sid --data_seed $seed5 --config 'configs/${config}.json' --L $numL --residual $residual --weight_decay $wd --epochs $epochs --bbp True --bbp_complexity $bbp_complexity --bbp_prior_sigma_1 $bbp_prior_sigma_1 & 
    python $code --dataset $dataset --gpu_id 6 --seed $sid --data_seed $seed6 --config 'configs/${config}.json' --L $numL --residual $residual --weight_decay $wd --epochs $epochs --bbp True --bbp_complexity $bbp_complexity --bbp_prior_sigma_1 $bbp_prior_sigma_1 & 
    python $code --dataset $dataset --gpu_id 7 --seed $sid --data_seed $seed7 --config 'configs/${config}.json' --L $numL --residual $residual --weight_decay $wd --epochs $epochs --bbp True --bbp_complexity $bbp_complexity --bbp_prior_sigma_1 $bbp_prior_sigma_1 & 
    wait" C-m
done


SEQ=$(seq 0 9)

for sid in $SEQ
do  

    dataset=HIV
    config=gin_hiv

    numL=4
    wd=0.
    residual=True
    epochs=200
    bbp_complexity=0.01
    bbp_prior_sigma_1=1.

    tmux send-keys "
    python $code --dataset $dataset --gpu_id 0 --seed $sid --data_seed $seed0 --config 'configs/${config}.json' --L $numL --residual $residual --weight_decay $wd --epochs $epochs --bbp True --bbp_complexity $bbp_complexity --bbp_prior_sigma_1 $bbp_prior_sigma_1 & 
    python $code --dataset $dataset --gpu_id 1 --seed $sid --data_seed $seed1 --config 'configs/${config}.json' --L $numL --residual $residual --weight_decay $wd --epochs $epochs --bbp True --bbp_complexity $bbp_complexity --bbp_prior_sigma_1 $bbp_prior_sigma_1 & 
    python $code --dataset $dataset --gpu_id 2 --seed $sid --data_seed $seed2 --config 'configs/${config}.json' --L $numL --residual $residual --weight_decay $wd --epochs $epochs --bbp True --bbp_complexity $bbp_complexity --bbp_prior_sigma_1 $bbp_prior_sigma_1 & 
    python $code --dataset $dataset --gpu_id 3 --seed $sid --data_seed $seed3 --config 'configs/${config}.json' --L $numL --residual $residual --weight_decay $wd --epochs $epochs --bbp True --bbp_complexity $bbp_complexity --bbp_prior_sigma_1 $bbp_prior_sigma_1 & 
    python $code --dataset $dataset --gpu_id 4 --seed $sid --data_seed $seed4 --config 'configs/${config}.json' --L $numL --residual $residual --weight_decay $wd --epochs $epochs --bbp True --bbp_complexity $bbp_complexity --bbp_prior_sigma_1 $bbp_prior_sigma_1 & 
    python $code --dataset $dataset --gpu_id 5 --seed $sid --data_seed $seed5 --config 'configs/${config}.json' --L $numL --residual $residual --weight_decay $wd --epochs $epochs --bbp True --bbp_complexity $bbp_complexity --bbp_prior_sigma_1 $bbp_prior_sigma_1 & 
    python $code --dataset $dataset --gpu_id 6 --seed $sid --data_seed $seed6 --config 'configs/${config}.json' --L $numL --residual $residual --weight_decay $wd --epochs $epochs --bbp True --bbp_complexity $bbp_complexity --bbp_prior_sigma_1 $bbp_prior_sigma_1 & 
    python $code --dataset $dataset --gpu_id 7 --seed $sid --data_seed $seed7 --config 'configs/${config}.json' --L $numL --residual $residual --weight_decay $wd --epochs $epochs --bbp True --bbp_complexity $bbp_complexity --bbp_prior_sigma_1 $bbp_prior_sigma_1 & 
    wait" C-m
done


SEQ=$(seq 0 9)

for sid in $SEQ
do  

    code=main_tox21.py
    dataset=HIV
    config=gin_hiv

    numL=4
    wd=0.
    residual=True
    epochs=200
    bbp_complexity=0.01
    bbp_prior_sigma_1=1.

    tmux send-keys "
    python $code --dataset $dataset --gpu_id 0 --seed $sid --data_seed $seed0 --config 'configs/${config}.json' --L $numL --residual $residual --weight_decay $wd --epochs $epochs --bbp True --bbp_complexity $bbp_complexity --bbp_prior_sigma_1 $bbp_prior_sigma_1 & 
    python $code --dataset $dataset --gpu_id 1 --seed $sid --data_seed $seed1 --config 'configs/${config}.json' --L $numL --residual $residual --weight_decay $wd --epochs $epochs --bbp True --bbp_complexity $bbp_complexity --bbp_prior_sigma_1 $bbp_prior_sigma_1 & 
    python $code --dataset $dataset --gpu_id 2 --seed $sid --data_seed $seed2 --config 'configs/${config}.json' --L $numL --residual $residual --weight_decay $wd --epochs $epochs --bbp True --bbp_complexity $bbp_complexity --bbp_prior_sigma_1 $bbp_prior_sigma_1 & 
    python $code --dataset $dataset --gpu_id 3 --seed $sid --data_seed $seed3 --config 'configs/${config}.json' --L $numL --residual $residual --weight_decay $wd --epochs $epochs --bbp True --bbp_complexity $bbp_complexity --bbp_prior_sigma_1 $bbp_prior_sigma_1 & 
    python $code --dataset $dataset --gpu_id 4 --seed $sid --data_seed $seed4 --config 'configs/${config}.json' --L $numL --residual $residual --weight_decay $wd --epochs $epochs --bbp True --bbp_complexity $bbp_complexity --bbp_prior_sigma_1 $bbp_prior_sigma_1 & 
    python $code --dataset $dataset --gpu_id 5 --seed $sid --data_seed $seed5 --config 'configs/${config}.json' --L $numL --residual $residual --weight_decay $wd --epochs $epochs --bbp True --bbp_complexity $bbp_complexity --bbp_prior_sigma_1 $bbp_prior_sigma_1 & 
    python $code --dataset $dataset --gpu_id 6 --seed $sid --data_seed $seed6 --config 'configs/${config}.json' --L $numL --residual $residual --weight_decay $wd --epochs $epochs --bbp True --bbp_complexity $bbp_complexity --bbp_prior_sigma_1 $bbp_prior_sigma_1 & 
    python $code --dataset $dataset --gpu_id 7 --seed $sid --data_seed $seed7 --config 'configs/${config}.json' --L $numL --residual $residual --weight_decay $wd --epochs $epochs --bbp True --bbp_complexity $bbp_complexity --bbp_prior_sigma_1 $bbp_prior_sigma_1 & 
    wait" C-m
done
