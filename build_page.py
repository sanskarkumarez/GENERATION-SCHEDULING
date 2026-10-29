"""Reads results/results.json and writes index.html, README.md and results/*.png (so every number shown comes from the run)."""
import json, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
R = json.load(open("results/results.json")); fm, pol, rf = R["forecast"], R["policy"], R["rf"]
INK, HZ, MU = "#12263A", "#F2C200", "#6B7C8C"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
# figures (for README)
f, ax = plt.subplots(figsize=(7, 3)); keys = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun", "holidays"]
for i, (m, c) in enumerate(zip(["naive", "arima", "sarima", "hybrid"], ["#B9C6D2", MU, INK, HZ])):
    ax.bar(np.arange(8) + (i-1.5)*.2, [fm[m][k]["MAPE"] for k in keys], .2, color=c, label=m)
ax.set_xticks(range(8)); ax.set_xticklabels(keys); ax.set_ylabel("MAPE (%)"); ax.legend(frameon=False, ncol=4); f.savefig("results/fig_forecast.png", dpi=170, bbox_inches="tight"); plt.close(f)
f, ax = plt.subplots(1, 2, figsize=(7, 2.8)); names = [k.split("  ")[0] for k in pol]
ax[0].bar(names, [v["cost"]/1e6 for v in pol.values()], color=[MU, INK, HZ, "#B9C6D2"]); ax[0].set_ylim(4800, 5050); ax[0].set_title("Operating cost (Rs million)")
ax[1].bar(names, [v["unserved"] for v in pol.values()], color=[MU, INK, HZ, "#B9C6D2"]); ax[1].set_title("Unserved energy (MWh)"); f.savefig("results/fig_policy.png", dpi=170, bbox_inches="tight"); plt.close(f)
f, ax = plt.subplots(figsize=(5, 2.6)); im = rf["importance"][::-1]; ax.barh([a for a, b in im], [b for a, b in im], color=INK); ax.set_title("Random Forest feature importance"); f.savefig("results/fig_importance.png", dpi=170, bbox_inches="tight"); plt.close(f)

A, B, C, D = [list(pol.values())[i] for i in range(4)]
pc = lambda a, b: (a-b)/b*100
S = dict(mape_a=fm["arima"]["all"]["MAPE"], mape_h=fm["hybrid"]["all"]["MAPE"], sun_a=fm["arima"]["Sun"]["MAPE"], sun_h=fm["hybrid"]["Sun"]["MAPE"],
         hol_a=fm["arima"]["holidays"]["MAPE"], hol_h=fm["hybrid"]["holidays"]["MAPE"], sar=fm["sarima"]["all"]["MAPE"], nai=fm["naive"]["all"]["MAPE"],
         cost_ca=pc(C["cost"], A["cost"]), cost_cb=pc(C["cost"], B["cost"]), cost_dc=pc(D["cost"], C["cost"]), uns_ca=pc(C["unserved"], A["unserved"]),
         uns_cb=pc(C["unserved"], B["unserved"]), uns_db=pc(D["unserved"], B["unserved"]), k=R["K"])
S["cost_ab"] = pc(B["cost"], A["cost"])
for k in ["cost_ab", "cost_ca", "uns_ca", "uns_cb"]: S[k + "_abs"] = abs(S[k])
S["book_c"] = "{:,}".format(round(C["booked"])); S["book_d"] = "{:,}".format(round(D["booked"]))
S["share"] = (B["unserved"]-C["unserved"])/(B["unserved"]-D["unserved"])*100
json.dump(S, open("results/summary.json", "w"), indent=1)
P = {"forecast": fm, "policy": {k.split("  ")[0]: dict(label=k, cost=v["cost"], unserved=v["unserved"], vh=v["viol_hours"], booked=v["booked"]) for k, v in pol.items()},
     "rf": rf, "sample": R["sample"]}
HTML = open("template.html").read().replace("__DATA__", json.dumps(P, separators=(",", ":")))
for k, v in S.items(): HTML = HTML.replace("{{%s}}" % k, ("%.1f" % v) if isinstance(v, float) else str(v))
for k, v in dict(auc="%.2f" % rf["auc"], rec="%d" % round(rf["recall"]*100), prec="%d" % round(rf["precision"]*100), flag="%.1f" % (rf["flagged_share"]*100)).items(): HTML = HTML.replace("{{%s}}" % k, v)
open("index.html", "w").write(HTML)
md = open("README_template.md").read()
for k, v in S.items(): md = md.replace("{{%s}}" % k, ("%.1f" % v) if isinstance(v, float) else str(v))
for k, v in dict(auc="%.2f" % rf["auc"], rec="%d" % round(rf["recall"]*100), prec="%d" % round(rf["precision"]*100), flag="%.1f" % (rf["flagged_share"]*100)).items(): md = md.replace("{{%s}}" % k, v)
open("README.md", "w").write(md); print(len(HTML), "bytes html", S)
