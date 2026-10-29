"""Optimal Scheduling of Mixed Energy Sources.
Pipeline: synthetic grid data -> day-ahead load forecast (ARIMA / SARIMA / K-means+ARIMA)
-> day-ahead economic dispatch (LP with ramp limits, hydro energy cap) -> real-time deviation
-> Random Forest flags hours likely to breach constraints -> targeted reserve policy.
All data is synthetic (see README). Run: python run_pipeline.py"""
import numpy as np, pandas as pd, json, warnings, time
from scipy.optimize import linprog
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, roc_auc_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
warnings.filterwarnings("ignore"); t0 = time.time()
rng = np.random.default_rng(7)
G = lambda h, m, s: np.exp(-0.5 * ((h - m) / s) ** 2)

# ---------------- 1. synthetic grid data (2023-2024, hourly) ----------------
days = pd.date_range("2023-01-01", "2024-12-31"); nd = len(days); H = np.arange(24)
HOL = ["2023-01-26","2023-03-08","2023-04-22","2023-08-15","2023-08-30","2023-09-07","2023-10-02","2023-10-24","2023-11-12",
       "2023-12-25","2024-01-26","2024-03-25","2024-04-11","2024-08-15","2024-08-19","2024-08-26","2024-10-02","2024-10-12",
       "2024-11-01","2024-12-25"]  # approximate festival / national holiday dates
hol = np.isin(days.strftime("%Y-%m-%d"), HOL); dow = days.dayofweek.values; doy = days.dayofyear.values
def ar1(n, phi, s):
    e = rng.normal(0, s, n); x = np.zeros(n)
    for i in range(1, n): x[i] = phi * x[i-1] + e[i]
    return x
Td = 26 + 9 * np.sin(2*np.pi*(doy-80)/365) + ar1(nd, .7, 1.2)            # daily mean temp (C)
quiet = (dow == 6) | hol
L = np.zeros((nd, 24)); noise = ar1(nd*24, .9, .012).reshape(nd, 24)
for d in range(nd):
    s = 0.55 + .12*G(H, 11, 3) + .30*G(H, 20.5, 2.4) if quiet[d] else 0.60 + .20*G(H, 10.5, 2.5) + .30*G(H, 20.5, 2.4)
    s = s * (0.88 if dow[d] == 6 else 0.84 if hol[d] else 0.96 if dow[d] == 5 else 1.0)
    cool = 1 + 0.018 * np.maximum(0, Td[d] + 5*np.sin(2*np.pi*(H-9)/24) - 27)
    ev = (35 + 50*d/nd) * (0.8 if quiet[d] else 1.0) * G(H, 21.5, 2.6)
    L[d] = 1000 * s * cool * (1 + noise[d]) + ev
clear = np.maximum(0, np.sin(np.pi*(H-6)/12))**1.3
base_c = 0.85 - 0.35 * ((doy > 180) & (doy < 270))
cd = np.clip(base_c + rng.normal(0, .12, nd), .15, 1); cf = np.clip(.55*cd + .45*base_c + rng.normal(0, .05, nd), .1, 1)
Sa = 500 * np.outer(cd, clear) * (1 + ar1(nd*24, .8, .04).reshape(nd, 24)); Sa = np.clip(Sa, 0, 500)
Sf = 500 * np.outer(cf, clear)
vs = 5.2 + 2.6*G(doy, 190, 55); dev = ar1(nd*24, .97, .35).reshape(nd, 24) * 2.4
v = np.clip(vs[:, None] + dev, 0, 30); vf = vs + .6*dev.mean(1) + rng.normal(0, .5, nd)
pc = lambda x: 350 * np.clip(((x-3)/8)**3, 0, 1)
Wa = pc(v); Wf = np.outer(pc(vf), np.ones(24))

# ---------------- 2. day-ahead load forecasting ----------------
E0 = int(np.where(days == "2024-03-01")[0][0]); T0 = int(np.where(days == "2024-09-01")[0][0])
import os, pickle
cal = np.c_[dow, hol, days.month.values, Td]
if os.path.exists("results/cache.pkl"):
    F, CL, best = pickle.load(open("results/cache.pkl", "rb"))
else:
    arima = [ARIMA(L[:E0, h], order=(2, 0, 1)).fit() for h in range(24)]
    sarima = [SARIMAX(L[:E0, h], order=(1, 0, 1), seasonal_order=(1, 0, 0, 7)).fit(disp=False) for h in range(24)]
    best = max(range(4, 9), key=lambda k: silhouette_score(L[:E0], KMeans(k, n_init=5, random_state=0).fit_predict(L[:E0])))
    km = KMeans(best, n_init=10, random_state=0).fit(L[:E0]); lab = km.labels_
    clf = RandomForestClassifier(200, random_state=0).fit(cal[:E0], lab)
    hy = {}; hv = {}
    for c in range(best):
        for h in range(24):
            ser = L[:E0][lab == c, h]; hv[(c, h)] = list(ser)
            hy[(c, h)] = ARIMA(ser, order=(1, 0, 1)).fit() if len(ser) >= 40 else None
    F = {k: np.zeros((nd, 24)) for k in ["naive", "arima", "sarima", "hybrid"]}; CL = np.zeros(nd, int)
    for d in range(E0, nd):
        c = int(clf.predict(cal[d:d+1])[0]); CL[d] = c
        for h in range(24):
            F["arima"][d, h] = arima[h].forecast(1)[0]; F["sarima"][d, h] = sarima[h].forecast(1)[0]
            F["hybrid"][d, h] = hy[(c, h)].forecast(1)[0] if hy[(c, h)] is not None else np.mean(hv[(c, h)])
        F["naive"][d] = L[d-7]
        ct = int(km.predict(L[d:d+1])[0])                                     # realised day -> update models
        for h in range(24):
            arima[h] = arima[h].extend([L[d, h]]); sarima[h] = sarima[h].extend([L[d, h]]); hv[(ct, h)].append(L[d, h])
            if hy[(ct, h)] is not None: hy[(ct, h)] = hy[(ct, h)].extend([L[d, h]])
    pickle.dump((F, CL, best), open("results/cache.pkl", "wb"))
print("forecasting done", round(time.time()-t0), "s; K =", best)
def met(f, a):
    e = f - a; return dict(MAPE=float(np.mean(np.abs(e)/a)*100), RMSE=float(np.sqrt(np.mean(e**2))), MAE=float(np.mean(np.abs(e))))
fm = {m: {"all": met(F[m][T0:], L[T0:]), **{n: met(F[m][T0:][dow[T0:] == i], L[T0:][dow[T0:] == i]) for i, n in enumerate(["Mon","Tue","Wed","Thu","Fri","Sat","Sun"])},
         "holidays": met(F[m][T0:][hol[T0:]], L[T0:][hol[T0:]])} for m in F}
for m in F: fm[m]["eve_bias"] = float(np.mean((F[m][T0:, 19:23] - L[T0:, 19:23]) / L[T0:, 19:23]) * 100)

# ---------------- 3. dispatch (LP), real-time deviation, costs ----------------
CAP = np.array([140,300,270,200.]); COST = np.array([300, 3000, 3500, 7000.]); RMP = np.array([80,50,45,100.]); HYD = 1700.
RESV, UNS, RCOST = 9000., 30000., 400.
def schedule(Lf, Sfc, Wfc, p0):
    T = 24; nx = 8*T; c = np.zeros(nx)
    for u in range(4): c[u*T:(u+1)*T] = COST[u]
    c[4*T:6*T] = 1; c[6*T:7*T] = 40000; c[7*T:] = 2000  # last block: over-generation absorbed (exports/backing down)
    Ae = np.zeros((T, nx))
    for t in range(T):
        for u in range(4): Ae[t, u*T+t] = 1
        Ae[t, 4*T+t] = -1; Ae[t, 5*T+t] = -1; Ae[t, 6*T+t] = 1; Ae[t, 7*T+t] = -1
    Au, bu = [], []
    for u in range(4):
        for t in range(T):
            r = np.zeros(nx); r[u*T+t] = 1
            if t: r[u*T+t-1] = -1
            Au.append(r); bu.append(RMP[u] + (0 if t else p0[u]))
            r2 = -r; r2 = np.zeros(nx); r2[u*T+t] = -1
            if t: r2[u*T+t-1] = 1
            Au.append(r2); bu.append(RMP[u] - (0 if t else p0[u]))
    r = np.zeros(nx); r[:T] = 1; Au.append(r); bu.append(HYD)
    bnd = [(0, CAP[u]) for u in range(4) for _ in range(T)] + [(0, s) for s in Sfc] + [(0, w) for w in Wfc] + [(0, None)]*(2*T)
    res = linprog(c, A_ub=np.array(Au), b_ub=bu, A_eq=Ae, b_eq=Lf-Sfc-Wfc, bounds=bnd, method="highs")
    assert res.x is not None, (np.isnan(Sfc).any(), np.isnan(Wfc).any(), p0, (Lf-Sfc-Wfc).min(), (Lf-Sfc-Wfc).max(), Sfc.max(), Wfc.max(), Wfc.shape); x = res.x; return x[:4*T].reshape(4, T), x[6*T:7*T], float((x[:4*T].reshape(4, T).sum(1) * COST).sum() + 40000*x[6*T:7*T].sum())
def realtime(p, Lf, Sfc, Wfc, La, Sac, Wac, R):
    cost = 0.; uns = np.zeros(24); act = np.zeros(24)
    for t in range(24):
        d = (La[t]-Sac[t]-Wac[t]) - (Lf[t]-Sfc[t]-Wfc[t])
        if d > 0:
            rem = d
            for u in (1, 2, 3):
                g = min(rem, max(0, min(CAP[u]-p[u, t], RMP[u]))); cost += g*COST[u]; rem -= g
            a = min(rem, R[t]); act[t] = a; cost += a*RESV; rem -= a; uns[t] = rem; cost += rem*UNS
        else:
            rem = -d
            for u in (3, 2, 1):
                s = min(rem, min(p[u, t], RMP[u])); cost -= s*COST[u]; rem -= s
    return cost, uns
P0 = np.array([100, 250, 150, 0.]); sch = {}
for m in ["arima", "hybrid"]:
    p0 = P0.copy(); sch[m] = {}
    for d in range(E0, nd):
        Lf_ = L[d] if m == "oracle" else F[m][d]
        Sf_, Wf_ = (Sa[d], Wa[d]) if m == "oracle" else (Sf[d], Wf[d])
        p, sh, c = schedule(Lf_, Sf_, Wf_, p0); sch[m][d] = (p, sh, c); p0 = p[:, 23]
print("LPs done", round(time.time()-t0), "s")

# ---------------- 4. violation labels + Random Forest ----------------
R0f = 0.02
rows = []
for d in range(E0, nd):
    p, sh, c = sch["hybrid"][d]; Lf_ = F["hybrid"][d]; net = Lf_ - Sf[d] - Wf[d]
    _, uns = realtime(p, Lf_, Sf[d], Wf[d], L[d], Sa[d], Wa[d], R0f*Lf_)
    hd = np.minimum(CAP[1:, None] - p[1:], RMP[1:, None]).sum(0)
    rec = np.abs(F["hybrid"][max(E0, d-7):d] - L[max(E0, d-7):d]).mean(0) if d > E0 else np.zeros(24)
    for t in range(24):
        rows.append(dict(day=d, hour=t, dow=dow[d], month=days[d].month, hol=int(hol[d]), Lf=Lf_[t], Sf=Sf[d][t], Wf=Wf[d][t], net=net[t],
                         ramp=net[t]-net[t-1] if t else 0, headroom=hd[t], hd_ratio=hd[t]/Lf_[t], gas=p[3, t], coalB=p[2, t],
                         cluster=CL[d], temp=Td[d], disagree=abs(F["hybrid"][d, t]-F["arima"][d, t]), recent_err=rec[t], viol=int(uns[t] > 0)))
df = pd.DataFrame(rows); feats = [c for c in df.columns if c not in ("day", "viol")]
tr, te = df[df.day < T0], df[df.day >= T0]
rf = RandomForestClassifier(300, min_samples_leaf=3, class_weight="balanced_subsample", oob_score=True, random_state=0, n_jobs=-1).fit(tr[feats], tr.viol)
oob = rf.oob_decision_function_[:, 1]
ths = np.linspace(.1, .8, 36); f2 = []
for th in ths:
    pr = oob > th; tp = ((pr) & (tr.viol == 1)).sum(); pp = tp/max(pr.sum(), 1); rc = tp/max(tr.viol.sum(), 1); f2.append(5*pp*rc/max(4*pp+rc, 1e-9))
thr = float(ths[int(np.argmax(f2))]); pte = rf.predict_proba(te[feats])[:, 1]; yte = te.viol.values
rfm = dict(auc=float(roc_auc_score(yte, pte)), precision=float(precision_score(yte, pte > thr, zero_division=0)), recall=float(recall_score(yte, pte > thr)),
           threshold=thr, base_rate=float(yte.mean()), flagged_share=float((pte > thr).mean()), n_train=int(len(tr)), n_test=int(len(te)),
           importance=sorted(zip(feats, map(float, rf.feature_importances_)), key=lambda z: -z[1])[:8])
df.loc[te.index, "p"] = pte

# ---------------- 5. policy comparison on held-out Sep-Dec ----------------
def run(policy, sched="hybrid", resv=None):
    tot = dict(cost=0., unserved=0., viol_hours=0, energy=0., booked=0., re_used=0.)
    for d in range(T0, nd):
        p, sh, c = sch[sched][d]; Lf_ = F[sched][d] if sched != "oracle" else L[d]
        R = resv(d, Lf_) if resv else R0f*Lf_
        Sx, Wx = (Sa[d], Wa[d]) if sched == "oracle" else (Sf[d], Wf[d])
        rc, uns = realtime(p, Lf_, Sx, Wx, L[d], Sa[d], Wa[d], R)
        tot["cost"] += c + rc + RCOST*R.sum(); tot["unserved"] += uns.sum() + sh.sum(); tot["viol_hours"] += int(((uns > 0) | (sh > 0)).sum())
        tot["energy"] += L[d].sum(); tot["booked"] += R.sum()
    tot["cost_per_mwh"] = tot["cost"]/tot["energy"]; return tot
pm = df.set_index(["day", "hour"]).p
pol = {"A  ARIMA forecast, 2% reserve": run("A", "arima"),
       "B  Hybrid forecast, 2% reserve": run("B", "hybrid"),
       "C  Hybrid + RF-targeted reserve": run("C", "hybrid", lambda d, Lf_: np.where(np.array([pm[(d, t)] for t in range(24)]) > thr, .08, R0f)*Lf_),
       "D  Hybrid + blanket 8% reserve": run("D", "hybrid", lambda d, Lf_: .08*Lf_)}
# ---------------- 6. outputs ----------------
worst = df[df.day >= T0].groupby("day").viol.sum().sort_values(ascending=False).index[:3].tolist()
pick = sorted(set(worst + [int(np.where(days == x)[0][0]) for x in ["2024-10-12", "2024-11-01", "2024-09-08", "2024-09-17", "2024-10-22", "2024-12-10", "2024-12-25", "2024-11-17"]]))
sample = {}
for d in pick:
    p, sh, c = sch["hybrid"][d]; dd = df[df.day == d]
    sample[str(days[d].date())] = dict(dow=int(dow[d]), hol=bool(hol[d]), actual=L[d].round().astype(int).tolist(), hybrid=F["hybrid"][d].round().astype(int).tolist(),
        arima=F["arima"][d].round().astype(int).tolist(), solar=Sa[d].round().astype(int).tolist(), wind=Wa[d].round().astype(int).tolist(),
        hydro=p[0].round().astype(int).tolist(), coal=(p[1]+p[2]).round().astype(int).tolist(), gas=p[3].round().astype(int).tolist(),
        viol=dd.viol.astype(int).tolist(), prob=(dd.p*100).round().astype(int).tolist())
json.dump(dict(K=best, forecast=fm, rf=rfm, policy=pol, sample=sample, test_days=int(nd-T0), eval_days=int(nd-E0), viol_rate=float(df.viol.mean()),
               runtime_s=round(time.time()-t0)), open("results/results.json", "w"), indent=1)
df.to_csv("results/hourly_features.csv", index=False)
print(json.dumps({k: v for k, v in dict(K=best, rf={k: v for k, v in rfm.items() if k != "importance"}, viol=float(df.viol.mean())).items()}, indent=1))
for m in fm: print(m, {k: round(v["MAPE"], 2) for k, v in fm[m].items() if k != "eve_bias"}, "eve_bias", round(fm[m]["eve_bias"], 2))
for k, v in pol.items(): print(k, {a: round(b, 1) for a, b in v.items()})
