```shell
tmux new -s dtqn
conda activate liuyang_dtqn_env
cd liuyang_project/pycharm_projects/dtqn
python run.py --project-name mobile-env --envs mobile-small-central-v0 --discount 0 --model DTQN --max-episode-steps 500 --tuf 15000 --dropout 0.2 --a-embed 16 --num-steps 3000000 --prepopulate 50000 --seed 3407 --device cuda --save-policy --verbose
```