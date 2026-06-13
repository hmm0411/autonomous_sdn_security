import pandas as pd
import matplotlib.pyplot as plt

dqn = pd.read_csv("logs/llm_auto_eval_dqn.csv")
ppo = pd.read_csv("logs/llm_auto_eval_ppo.csv")

summary = pd.DataFrame({
    "model":["DQN","PPO"],

    "accuracy":[
        dqn["qual_accuracy"].mean(),
        ppo["qual_accuracy"].mean()
    ],

    "clarity":[
        dqn["qual_clarity"].mean(),
        ppo["qual_clarity"].mean()
    ],

    "usefulness":[
        dqn["qual_usefulness"].mean(),
        ppo["qual_usefulness"].mean()
    ]
})

summary.set_index("model").plot(
    kind="bar",
    figsize=(8,5)
)

plt.ylabel("Score (1-5)")
plt.title("LLM Qualitative Evaluation")
plt.tight_layout()

plt.savefig(
    "results/evaluation/llm_qualitative_compare.png"
)

print(summary)
