import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("logs/llm_auto_eval_dqn_small.csv")

metrics = {
    "Qual Accuracy": df.groupby("scenario")["qual_accuracy"].mean(),
    "Qual Clarity": df.groupby("scenario")["qual_clarity"].mean(),
    "Qual Usefulness": df.groupby("scenario")["qual_usefulness"].mean(),
}

plot_df = pd.DataFrame(metrics)

plot_df.plot(kind="bar")
plt.ylabel("Score")
plt.title("LLM Qualitative Evaluation")
plt.tight_layout()

plt.savefig("results/evaluation/llm_qualitative.png")
