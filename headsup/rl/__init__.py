"""rl_games integration: env registration, PPO exploiter CLI, ONNX export.

    python -m headsup.rl.exploitability -f configs/rl_games_exploiter.yaml -t --opponent cfr
    python -m headsup.rl.onnx -f configs/rl_games_exploiter.yaml -m runs/<exp>/nn/exploitability.pth -o out.onnx
"""
