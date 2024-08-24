# headsup-poker
Heads-Up poker openai gym environment for fun 

## Visualization and demo

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

### Evaluation
`python exploitability.py -f rl_config/poker_env.yaml -p -c runs/<exp_folder>/<checkpoint name>.pth`