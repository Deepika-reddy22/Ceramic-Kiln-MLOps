import numpy as np, pandas as pd, os
rng=np.random.default_rng(42); n=300
load=rng.uniform(15,180,n); temp=rng.uniform(750,1250,n); moisture=rng.uniform(2,18,n)
duration=rng.uniform(5,18,n); insulation=rng.uniform(.65,.98,n); volume=rng.uniform(.3,2.5,n)
ambient=rng.uniform(15,40,n)
energy=np.maximum(.018*load*(temp-25)/100+1.7*duration+16*load*moisture/100+
22*(load*temp*duration/(volume*1000))*(1.05-insulation)+7*volume+.22*(25-ambient)+rng.normal(0,4,n),5)
base=pd.DataFrame({"pottery_load_kg":load,"target_temperature_c":temp,"initial_moisture_pct":moisture,
"planned_duration_hr":duration,"insulation_rating":insulation,"kiln_volume_m3":volume,
"ambient_temperature_c":ambient,"energy_consumption_kwh":energy})
os.makedirs("data",exist_ok=True)
v1=base.copy(); v1.insert(0,"batch_id",[f"V1-{i+1:04d}" for i in range(n)])
v2=base.copy(); v2["insulation_rating"]=np.clip(v2.insulation_rating-rng.uniform(.12,.22,n),.35,.85)
v2["energy_consumption_kwh"]=np.maximum(v2.energy_consumption_kwh*1.18+28*(1-v2.insulation_rating)+rng.normal(0,3,n),5)
v2.insert(0,"batch_id",[f"V2-{i+1:04d}" for i in range(n)])
v1.to_csv("data/kiln_batches_v1.csv",index=False); v2.to_csv("data/kiln_batches_v2.csv",index=False)
