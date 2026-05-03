modes = ["no_defense", "rule", "rl", "rl_twin"]

for mode in modes:
    print("Running:", mode)
    os.environ["MODE"] = mode
    os.system("python control_loop/main_loop.py")