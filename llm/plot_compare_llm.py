import pandas as pd
import matplotlib.pyplot as plt
import os

os.makedirs("results/evaluation", exist_ok=True)

dqn = pd.read_csv("logs/llm_auto_eval_dqn.csv")
ppo = pd.read_csv("logs/llm_auto_eval_ppo.csv")

summary = pd.DataFrame({
    "model":["DQN","PPO"],

    "accuracy":[
        dqn["exact_match"].mean()*100,
        ppo["exact_match"].mean()*100
    ],

    "faithfulness":[
        dqn["faithfulness_rule"].mean(),
        ppo["faithfulness_rule"].mean()
    ],

    "latency":[
        dqn["latency_ms"].mean(),
        ppo["latency_ms"].mean()
    ],

    "relevancy":[
        dqn["answer_relevancy"].mean(),
        ppo["answer_relevancy"].mean()
    ]
})

# Accuracy
plt.figure(figsize=(6,4))
plt.bar(summary["model"], summary["accuracy"])
plt.ylabel("Accuracy (%)")
plt.title("LLM Strict Accuracy")
plt.tight_layout()
plt.savefig("results/evaluation/llm_accuracy.png")
plt.close()

# Latency
plt.figure(figsize=(6,4))
plt.bar(summary["model"], summary["latency"])
plt.ylabel("Latency (ms)")
plt.title("LLM Response Latency")
plt.tight_layout()
plt.savefig("results/evaluation/llm_latency.png")
plt.close()

# Relevancy
plt.figure(figsize=(6,4))
plt.bar(summary["model"], summary["relevancy"])
plt.ylabel("Relevancy")
plt.title("LLM Answer Relevancy")
plt.tight_layout()
plt.savefig("results/evaluation/llm_relevancy.png")
plt.close()

# Faithfulness
plt.figure(figsize=(6,4))
plt.bar(summary["model"], summary["faithfulness"])
plt.ylabel("Faithfulness")
plt.title("LLM Faithfulness")
plt.tight_layout()
plt.savefig("results/evaluation/llm_faithfulness.png")
plt.close()

print(summary)
