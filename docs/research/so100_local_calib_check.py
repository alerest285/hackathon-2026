"""Model-free calibration tests: our native SO-100 demos vs the checkpoint's reference data (VoicAndrei/so100_kitchen,
old LeRobot convention, whose q01/q99 the checkpoint ships). No GPU.
T1 signs: joint-velocity correlations between body joints must have the same signs in both datasets once mapped.
T2 offsets/scale: overlap of our mapped poses with kitchen poses, in the checkpoint's normalized space."""
import itertools, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.spatial import cKDTree

ours_root, kitchen_root, qmap_path = map(Path, sys.argv[1:4])
J = ["pan", "lift", "elbow", "wflex", "wroll"]
Q01 = np.array([-92.021484375, 84.814453125, 61.435546875, -23.994140625, -124.541015625])
Q99 = np.array([54.140625, 184.21875, 178.857421875, 80.595703125, 60.205078125])
SCALE = 4095 / 4096
LAG = 5  # frames (~167 ms) for velocities

def load(files):
    d = pd.concat([pd.read_parquet(f, columns=["observation.state", "episode_index", "frame_index"]) for f in files])
    d = d.sort_values(["episode_index", "frame_index"])
    return np.stack(d["observation.state"].to_numpy())[:, :5].astype(float), d["episode_index"].to_numpy()

S_o, E_o = load(sorted((ours_root / "data").rglob("*.parquet")))
S_k, E_k = load(sorted((kitchen_root / "data").glob("*.parquet")))
print(f"ours {len(np.unique(E_o))} eps {len(S_o)} frames | kitchen sample {len(np.unique(E_k))} eps {len(S_k)} frames")

def velocities(S, E):
    same = E[LAG:] == E[:-LAG]
    V = (S[LAG:] - S[:-LAG])[same]
    return V, E[LAG:][same]

def corr(V):
    m = np.linalg.norm(V, axis=1) > 3.0  # moving frames only
    return np.corrcoef(V[m].T)

V_o, VE_o = velocities(S_o, E_o); V_k, VE_k = velocities(S_k, E_k)
C_o, C_k = corr(V_o), corr(V_k)
rng = np.random.default_rng(0)
def boot(V, VE, n=200):
    eps = np.unique(VE); idx = {e: np.flatnonzero(VE == e) for e in eps}
    return np.stack([corr(V[np.concatenate([idx[e] for e in rng.choice(eps, len(eps))])]) for _ in range(n)])
B_o, B_k = boot(V_o, VE_o), boot(V_k, VE_k)

print("\nT1. joint-velocity correlations (moving frames, lag 5), [95% CI over episodes]")
print(f"{'pair':14s} {'kitchen (old conv)':>24s} {'ours (native)':>24s}")
pairs = list(itertools.combinations(range(5), 2))
for i, j in pairs:
    ck, co = C_k[i, j], C_o[i, j]
    lk, hk = np.percentile(B_k[:, i, j], [2.5, 97.5]); lo, ho = np.percentile(B_o[:, i, j], [2.5, 97.5])
    print(f"{J[i]+'-'+J[j]:14s} {ck:+.2f} [{lk:+.2f},{hk:+.2f}]      {co:+.2f} [{lo:+.2f},{ho:+.2f}]")

# score every sign vector (pan fixed +1: a global flip leaves correlations unchanged; pan is hand-checked)
def score(s, Co=C_o):
    return sum(s[i] * s[j] * Co[i, j] * C_k[i, j] for i, j in pairs)
res = []
for rest in itertools.product([1, -1], repeat=4):
    s = np.array((1,) + rest)
    res.append((score(s), s, np.mean([score(s, b) for b in B_o[:50]])))
res.sort(key=lambda r: -r[0])
print("\nsign vectors ranked by agreement sum_ij s_i s_j C_ours C_kitchen (pan fixed +1):")
for sc, s, _ in res[:6]:
    tag = "  <- T21 / qmap signs" if list(s) == [1, -1, 1, 1, -1] else ""
    print(f"  {s.astype(int).tolist()}  {sc:+.3f}{tag}")
# per-joint: flipping only that joint from T21 signs
t21 = np.array([1, -1, 1, 1, -1])
print("flip one joint from T21 signs -> score change (negative = T21 sign better):")
for i in range(1, 5):
    s = t21.copy(); s[i] = -s[i]
    d = np.array([score(s, b) - score(t21, b) for b in B_o])
    print(f"  {J[i]:6s} {score(s) - score(t21):+.3f}  [95% CI {np.percentile(d, 2.5):+.3f}, {np.percentile(d, 97.5):+.3f}]")

# T2: candidate maps, model = A * ours + B (body joints)
def first(S, E):
    return np.stack([S[E == e][0] for e in np.unique(E)])
rest_o, rest_k = np.median(first(S_o, E_o), 0), np.median(first(S_k, E_k), 0)
print("\nrest (median episode start) ours", np.round(rest_o, 1), " kitchen", np.round(rest_k, 1))
q = json.loads(qmap_path.read_text())
qa = (np.array(q["model_q99"]) - np.array(q["model_q01"])) / (np.array(q["ours_q99"]) - np.array(q["ours_q01"]))
qb = np.array(q["model_q01"]) - qa * np.array(q["ours_q01"])
def phys(sign):
    A = np.array(sign, float) * SCALE
    return A, rest_k - A * rest_o
cands = {"qmap (T21 signs, quantile scale)": (qa[:5], qb[:5]), "phys (T21 signs, |k|=1, rest)": phys(t21)}
for i in range(1, 5):
    s = t21.copy(); s[i] = -s[i]; cands[f"phys, {J[i]} flipped"] = phys(s)
A, Bq = qa[:5].copy(), qb[:5].copy(); mid = (q["ours_q01"][4] + q["ours_q99"][4]) / 2; A[4] = -qa[4]
Bq[4] = (qa[4] * mid + qb[4]) - A[4] * mid  # same span, reversed about the data midpoint
cands["qmap, wroll flipped"] = (A, Bq)
norm = lambda X: 2 * (X - Q01) / (Q99 - Q01) - 1
K = norm(S_k); tree = cKDTree(K)
sub = rng.choice(len(S_o), 8000, replace=False)
print("\nT2. our poses mapped into the checkpoint's normalized space (kitchen q01..q99 = [-1, 1])")
print(f"{'candidate':36s} {'NN dist to kitchen p50/p90':>27s}  per-joint inside [-1,1] %  (pan lift elbow wflex wroll)")
for name, (A, Bv) in cands.items():
    X = norm(A * S_o[sub] + Bv)
    dist, _ = tree.query(X)
    inside = (np.abs(X) <= 1).mean(0) * 100
    print(f"{name:36s} {np.median(dist):12.2f} / {np.percentile(dist, 90):5.2f}       {'  '.join(f'{v:4.0f}' for v in inside)}")
print("\nper-joint mapped range of OUR data, normalized p1..p99:")
for name, (A, Bv) in list(cands.items())[:2]:
    X = norm(A * S_o + Bv)
    print(f"  {name:36s}", "  ".join(f"{J[i]} {np.percentile(X[:, i], 1):+.1f}..{np.percentile(X[:, i], 99):+.1f}" for i in range(5)))
Xk = norm(S_k)
print(f"  {'kitchen itself':36s}", "  ".join(f"{J[i]} {np.percentile(Xk[:, i], 1):+.1f}..{np.percentile(Xk[:, i], 99):+.1f}" for i in range(5)))
