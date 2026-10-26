import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
df=pd.read_csv("table_all_methods_per_seed.csv")
CLS=["Normal","DDoS","Flow Overflow","IP Spoofing","Packet-In","Port Scan"]
M=[("Rule","#9e9e9e"),("DQN","#1f77b4"),("PPO","#ff7f0e")]
fig,ax=plt.subplots(figsize=(6.6,3.0)); w=0.26; xs=np.arange(len(CLS))
for i,(m,c) in enumerate(M):
    sub=df[df.method==m]; vals=sub[[f"acc_{n}" for n in CLS]].values*100
    mean=vals.mean(0); sd=vals.std(0,ddof=1) if len(sub)>1 else np.zeros(len(CLS))
    lo=np.minimum(sd,mean); hi=np.minimum(sd,100-mean)
    x=xs+(i-1)*w
    ax.bar(x,mean,w,color=c,label=f"{m}" + (f" (n={len(sub)})" if len(sub)>1 else ""),
           yerr=[lo,hi] if len(sub)>1 else None,capsize=2,error_kw={"lw":0.8})
    if len(sub)>1:
        for k in range(len(CLS)):
            ax.scatter(np.full(len(sub),x[k])+np.linspace(-0.07,0.07,len(sub)),vals[:,k],s=6,color="black",zorder=3)
ax.set_xticks(xs); ax.set_xticklabels(CLS,fontsize=8); ax.set_ylim(0,108)
ax.set_ylabel("Acceptable-action rate (%)",fontsize=8); ax.tick_params(axis="y",labelsize=8)
ax.legend(fontsize=7,ncol=3,loc="upper center",bbox_to_anchor=(0.5,1.16),frameon=False)
ax.grid(axis="y",alpha=0.3); fig.tight_layout(); fig.savefig("fig_test_per_class.png",dpi=300)
