"""Make reproducible demonstration forecasts from a simulated latent predictor."""
import numpy as np
import pandas as pd
from pathlib import Path
rng=np.random.default_rng(13)
rows=[]
for site in "ABCDE":
    site_shift=rng.normal(scale=.25)
    for month in [11,12,1,2]:
        year=2023 if month>=11 else 2024
        for day in range(1,12):
            latent=rng.normal()+site_shift
            actual_prob=1/(1+np.exp(-(-1.3+1.4*latent)))
            event=int(rng.random()<actual_prob)
            baseline=float(np.clip(.20 + 0.07*site_shift,.01,.99))
            noisy_signal=latent+rng.normal(scale=.6)
            candidate=float(1/(1+np.exp(-(-1.3+1.0*noisy_signal))))
            rows.append((f"{year}-{month:02d}-{day:02d}",site,event,round(baseline,4),round(candidate,4)))
out=Path(__file__).parent/"synthetic_forecasts.csv"
pd.DataFrame(rows,columns=["date","site","event","baseline","candidate"]).to_csv(out,index=False)
print(out)
