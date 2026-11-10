"""Export trained weights and team form to web/model.json for the browser app.

Usage:  python export_web.py web/model.json web/reference_predictions.json
"""
import sys, json
import joblib, numpy as np, pandas as pd
from pathlib import Path
from features import form_columns, matchup_row
b=joblib.load('models/models.joblib')
m=b['models']; out={'features':b['features'],'form_cols':form_columns(),'data_through':b['data_through'],'models':{}}
for k,v in m.items():
    if not isinstance(v,dict): out[k]=v; continue
    pipe=v['model']; sc,est=pipe[0],pipe[1]
    coef=np.ravel(est.coef_); icpt=float(np.ravel(est.intercept_)[0])
    # fold scaler into raw-feature weights
    w=coef/sc.scale_; c=icpt-float((coef*sc.mean_/sc.scale_).sum())
    out['models'][k]={'w':[round(float(x),8) for x in w],'b':round(c,8),**({'sd':round(v['resid_sd'],4)} if 'resid_sd' in v else {})}
f=b['team_form']; out['teams']={t:[round(float(f.at[t,c]),5) for c in form_columns()] for t in f.index}
out['metrics']=json.load(open('models/metrics.json'))
# Upcoming games for the next two weeks, with each team's days of rest before that game.
if Path('data/schedule.csv').exists():
    ls=pd.read_csv('data/line_scores.csv'); sched=pd.read_csv('data/schedule.csv')
    through=pd.Timestamp(b['data_through']); horizon=through+pd.Timedelta(days=15)
    sched['d']=pd.to_datetime(sched['date'])
    days=pd.concat([pd.DataFrame({'team':ls[s],'d':pd.to_datetime(ls['date'])}) for s in ('home','away')]+
                   [pd.DataFrame({'team':sched[s],'d':sched['d']}) for s in ('home','away')]).drop_duplicates()
    def rest(team,d):
        prev=days[(days.team==team)&(days.d<d)].d.max()
        return 4 if pd.isna(prev) else int(min(4,max(1,(d-prev).days)))
    up=sched[(sched.d>through)&(sched.d<=horizon)&sched.home.isin(f.index)&sched.away.isin(f.index)]
    out['upcoming']=[{'date':r.date,'tip':r.tipoff_utc.replace(' ','T'),'home':r.home,'away':r.away,'hr':rest(r.home,r.d),'ar':rest(r.away,r.d)} for r in up.itertuples()]
json.dump(out,open(sys.argv[1],'w'),separators=(',',':'))
# reference predictions
ref=[]
for h,a,hr,ar in [('LAL','BOS',2,2),('OKC','DET',1,3),('CHA','NYK',4,1)]:
    row=matchup_row(f,h,a,hr,ar)[b['features']]
    ref.append({'h':h,'a':a,'hr':hr,'ar':ar,**{k:float(m[k]['model'].predict(row)[0]) for k in m if k.startswith(('spread','total'))},**{k:float(m[k]['model'].predict_proba(row)[0,1]) for k in m if k.startswith('win')}})
json.dump(ref,open(sys.argv[2],'w'))
print(len(out['features']), len(out['teams']), list(out['models'])[:3])
