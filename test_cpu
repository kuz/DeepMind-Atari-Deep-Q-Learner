#!/bin/bash

if [ -z "$1" ]
  then echo "Please provide the name of the game, e.g.  ./watch_pretrained breakout"; exit 0
fi

if [ -z "$2" ]
  then echo "Please provide the pretrained network file, e.g.  ./watch_pretrained breakout DQN3_0_1_breakout_FULL_Y.t7"; exit 0
fi

ENV=$1
NETWORK=$2
FRAMEWORK="alewrap"

game_path=$PWD"/roms/"
env_params="useRGB=true"
agent="NeuralQLearner"
n_replay=1
netfile="\"convnet_atari3\""
update_freq=4
actrep=4
discount=0.99
seed=1
learn_start=50000
pool_frms_type="\"max\""
pool_frms_size=2
initial_priority="false"
replay_memory=1000000
eps_end=0.1
eps_endt=replay_memory
lr=0.00025
agent_type="DQN3_0_1"
preproc_net="\"net_downsample_2x_full_y\""
agent_name=$agent_type"_"$1"_FULL_Y"
state_dim=7056
ncols=1
agent_params="lr="$lr",ep=1,ep_end="$eps_end",ep_endt="$eps_endt",discount="$discount",hist_len=4,learn_start="$learn_start",replay_memory="$replay_memory",update_freq="$update_freq",n_replay="$n_replay",network="$netfile",preproc="$preproc_net",state_dim="$state_dim",minibatch_size=32,rescale_r=1,ncols="$ncols",bufferSize=512,valid_size=500,target_q=10000,clip_delta=1,min_reward=-1,max_reward=1"
gif_file="../gifs/$ENV.gif"
gpu=-1
random_starts=30
pool_frms="type="$pool_frms_type",size="$pool_frms_size
num_threads=4

args="-framework $FRAMEWORK -game_path $game_path -name $agent_name -env $ENV -env_params $env_params -agent $agent -agent_params $agent_params -actrep $actrep -gpu $gpu -random_starts $random_starts -pool_frms $pool_frms -seed $seed -threads $num_threads -network $NETWORK -gif_file $gif_file"
echo $args

cd dqn
../torch/bin/qlua test_agent.lua $args
