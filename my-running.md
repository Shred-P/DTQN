


```shell
python run.py --envs gv_memory.7x7.yaml --in-embed 128 --num-steps 2000000 --save-policy --verbose
```

```shell
python run.py --envs gv_memory.7x7.yaml --pos sin --disable-wandb  --render --verbose
```

/Users/camille/PycharmProjects/DTQN-main/policies/DTQN-test/gv_memory.7x7.yaml/model=DTQN_envs=gv_memory.7x7.yaml_obs_embed=8_a_embed=0_in_embed=128_context=50_heads=8_layers=2_batch=32_gate=res_identity=False_history=50_pos=learned_bag=0_seed=1_checkpoint.pt'
     
```shell
python run.py --envs gv_memory.7x7.yaml --in-embed 128 --pos sin --num-steps 2000000 --save-policy --verbose
```        

    
```shell
python run.py --envs gv_memory.7x7.yaml --model DQN --num-steps 1000000 --save-policy --verbose
```        

```shell
python run.py --envs gv_memory.5x5.yaml --in-embed 128 --pos learned --num-steps 200000 --save-policy --verbose
```        
```shell
python run.py --envs gv_memory.7x7.yaml --in-embed 128 --bag-size 10 --pos learned --num-steps 1500000 --save-policy --verbose
```        
