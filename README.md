# headsup-poker
Heads-Up poker openai gym environment for fun. DeepCFR training and evaluation.  

## Visualization and demo

`python visualize.py`

![Screenshot](https://github.com/romaf5/headsup-poker/blob/main/imgs/headsup-test.jpg?raw=true)


## DeepCFR training 

`cd deepcfr; python train_deepcfr.py`

## RLGames exploitability evaluation

Add gym environment registration to `rl_games/common/env_configurations.py`

```
gym.register(id="HeadsUpPokerRLGames-v0", entry_point="rl_games_env:HeadsUpPokerRLGames")
```

### Training
`python exploitability.py -f rl_config/poker_env.yaml`

### Evaluation (current model: 72-100 mbb/g)
`python exploitability.py -f rl_config/poker_env.yaml -p -c runs/<exp_folder>/<checkpoint name>.pth`

### Model conversion to onnx
`python rl_games_onnx.py -f rl_config/poker_env.yaml -m runs/<exp_folder>/<checkpoint name>.pth`

### Play against rl-games model
Uncomment `player = ONNXRLGamesPlayer()` in `visualize.py`, use `python visualize.py` to play against bot