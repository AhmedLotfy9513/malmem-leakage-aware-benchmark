#!/usr/bin/env python3
import argparse, hashlib, json, os, platform, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, matthews_corrcoef, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None
try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None
try:
    from catboost import CatBoostClassifier
except Exception:
    CatBoostClassifier = None

SEED = 20260911

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def family_name(v):
    s=str(v).strip()
    if 'benign' in s.lower(): return 'Benign'
    for sep in ['-','_',':']:
        p=s.split(sep)
        if len(p)>=2 and p[0].lower() in {'ransomware','spyware','trojan','trojanhorse'}:
            return p[0]+'-'+p[1]
    return s

def load_data(csv_path,out):
    df=pd.read_csv(csv_path)
    low={c.lower():c for c in df.columns}
    cls=low.get('class',low.get('label'))
    cat=low.get('category')
    if cls is None: raise ValueError('Need Class or Label column')
    yb=df[cls].astype(str).str.lower().map(lambda s:'Benign' if ('benign' in s or s in {'0','false'}) else 'Malware')
    yf=df[cat].map(family_name) if cat else None
    drop=[cls]+([cat] if cat else [])
    X=df.drop(columns=drop,errors='ignore').select_dtypes(include=[np.number]).replace([np.inf,-np.inf],np.nan)
    if X.shape[1]==0: raise ValueError('No numeric predictors found')
    groups=pd.util.hash_pandas_object(X,index=False).astype(str).to_numpy()
    audit={
      'sha256':sha256(csv_path),'rows':len(df),'numeric_features':X.shape[1],
      'features':list(X.columns),'binary_counts':yb.value_counts().to_dict(),
      'family_counts':None if yf is None else yf.value_counts().to_dict(),
      'exact_profile_groups':int(pd.Series(groups).nunique()),
      'warning':'Use real specimen/run/host groups when available; exact-profile grouping is only a fallback.'
    }
    (out/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf8')
    if yf is not None: yf.value_counts().rename_axis('family').reset_index(name='count').to_csv(out/'family_counts.csv',index=False)
    return X,yb,yf,groups

def outer_split(y,groups,seed):
    enc=LabelEncoder().fit_transform(y)
    cv=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=seed)
    return next(cv.split(np.zeros(len(enc)),enc,groups))

def mi(X,y): return mutual_info_classif(X,y,random_state=SEED)

def selector(rep,k):
    if rep=='full': return 'passthrough'
    if rep=='anova': return SelectKBest(f_classif,k=k)
    if rep=='mi': return SelectKBest(mi,k=k)
    if rep=='pca': return PCA(n_components=k,random_state=SEED)
    raise ValueError(rep)

def models(seed,nc):
    d={
      'dummy':(DummyClassifier(strategy='prior'),{}),
      'rf':(RandomForestClassifier(random_state=seed,n_jobs=-1,class_weight='balanced_subsample'),{
        'clf__n_estimators':[300,700],'clf__max_depth':[None,20],'clf__min_samples_leaf':[1,2]}),
      'svc':(SVC(kernel='rbf',class_weight='balanced'),{'clf__C':[0.1,1,10],'clf__gamma':['scale',0.01]}),
      'hgb':(HistGradientBoostingClassifier(random_state=seed),{'clf__learning_rate':[0.05,0.1],'clf__max_leaf_nodes':[15,31]})
    }
    if XGBClassifier:
      kw=dict(tree_method='hist',random_state=seed,n_jobs=-1,eval_metric='logloss' if nc==2 else 'mlogloss')
      if nc>2: kw.update(objective='multi:softprob',num_class=nc)
      d['xgb']=(XGBClassifier(**kw),{'clf__n_estimators':[300,700],'clf__max_depth':[4,7],'clf__learning_rate':[0.03,0.08]})
    if LGBMClassifier: d['lgbm']=(LGBMClassifier(random_state=seed,n_jobs=-1,verbose=-1),{'clf__n_estimators':[300,700],'clf__num_leaves':[15,31]})
    if CatBoostClassifier: d['catboost']=(CatBoostClassifier(random_seed=seed,verbose=False,allow_writing_files=False),{'clf__iterations':[300,700],'clf__depth':[5,8]})
    return d

def pipeline(model,sel,balance):
    steps=[('impute',SimpleImputer(strategy='median',keep_empty_features=True)),('scale',StandardScaler()),('select',sel)]
    if balance=='smote':
        steps += [('smote',SMOTE(random_state=SEED)),('clf',model)]
        return ImbPipeline(steps)
    steps += [('clf',model)]
    return SkPipeline(steps)

def benign_fpr(yt,yp,b):
    m=(yt==b)
    return float(np.mean(yp[m]!=b)) if m.sum() else float('nan')

def metrics(yt,yp,b):
    p,r,f,_=precision_recall_fscore_support(yt,yp,average='macro',zero_division=0)
    return {'accuracy':accuracy_score(yt,yp),'macro_precision':p,'macro_recall':r,'macro_f1':f,'mcc':matthews_corrcoef(yt,yp),'benign_fpr':benign_fpr(yt,yp,b)}

def timing(est,X):
    one=X.iloc[[0]]
    for _ in range(10): est.predict(one)
    a=[]
    for _ in range(100):
        t=time.perf_counter_ns(); est.predict(one); a.append((time.perf_counter_ns()-t)/1e6)
    return {'single_p50_ms':float(np.percentile(a,50)),'single_p95_ms':float(np.percentile(a,95))}

def run_task(X,ystr,groups,out,name,seed):
    out.mkdir(parents=True,exist_ok=True)
    le=LabelEncoder(); y=le.fit_transform(ystr)
    tr,te=outer_split(ystr,groups,seed)
    pd.DataFrame({'row_id':np.arange(len(X)),'split':np.where(np.isin(np.arange(len(X)),tr),'development','test'),'group':groups,'target':ystr}).to_csv(out/'split.csv',index=False)
    Xtr,Xte=X.iloc[tr],X.iloc[te]; ytr,yte=y[tr],y[te]; gtr=groups[tr]
    inner=StratifiedGroupKFold(n_splits=4,shuffle=True,random_state=seed+1)
    rows=[]; winner=None
    budgets=[7,15,25,X.shape[1]]
    for rep in ['full','anova','mi','pca']:
      for k in budgets:
        if rep=='full' and k!=X.shape[1]: continue
        if k>X.shape[1]: continue
        sel=selector(rep,k)
        for mn,(m,grid) in models(seed,len(le.classes_)).items():
          for bal in ['none','smote']:
            if mn=='dummy' and bal=='smote': continue
            est=pipeline(clone(m),clone(sel) if sel!='passthrough' else sel,bal)
            gs=GridSearchCV(est,grid,scoring='f1_macro',cv=inner,n_jobs=-1,refit=True,error_score='raise')
            t=time.perf_counter(); gs.fit(Xtr,ytr,groups=gtr); train_s=time.perf_counter()-t
            pred=gs.predict(Xte)
            b=int(np.where(le.classes_=='Benign')[0][0]) if 'Benign' in le.classes_ else 0
            rec={'task':name,'representation':rep,'k':int(k),'model':mn,'balancing':bal,'cv_macro_f1':float(gs.best_score_),'train_seconds':train_s,**metrics(yte,pred,b),**timing(gs.best_estimator_,Xte),'best_params':json.dumps(gs.best_params_,sort_keys=True)}
            rows.append(rec)
            if winner is None or rec['cv_macro_f1']>winner['record']['cv_macro_f1']:
                winner={'record':rec,'pred':pred.copy(),'yte':yte.copy(),'te':te.copy(),'labels':list(le.classes_)}
    pd.DataFrame(rows).to_csv(out/'summary_metrics.csv',index=False)
    (out/'winner.json').write_text(json.dumps(winner['record'],indent=2),encoding='utf8')
    pd.DataFrame({'row_id':winner['te'],'y_true':[le.classes_[i] for i in winner['yte']],'y_pred':[le.classes_[i] for i in winner['pred']]}).to_csv(out/'winner_predictions.csv',index=False)
    fig,ax=plt.subplots(figsize=(8,7)); ConfusionMatrixDisplay.from_predictions(winner['yte'],winner['pred'],display_labels=winner['labels'],xticks_rotation=90,ax=ax); fig.tight_layout(); fig.savefig(out/'winner_confusion_matrix.pdf',bbox_inches='tight'); plt.close(fig)

def run_lofo(X,yb,yf,groups,out,seed):
    if yf is None or XGBClassifier is None: return
    out.mkdir(parents=True,exist_ok=True); rows=[]
    for fam in sorted(f for f in yf.unique() if f!='Benign'):
        held=(yf.to_numpy()==fam); candidates=np.where(~held)[0]
        trloc,teloc=outer_split(yb.iloc[candidates],groups[candidates],seed)
        dev=candidates[trloc]; pool=candidates[teloc]
        benign=pool[yb.iloc[pool].to_numpy()=='Benign']; test=np.r_[np.where(held)[0],benign]
        le=LabelEncoder().fit(['Benign','Malware']); ytr=le.transform(yb.iloc[dev]); yte=le.transform(yb.iloc[test])
        est=SkPipeline([('impute',SimpleImputer(strategy='median',keep_empty_features=True)),('scale',StandardScaler()),('clf',XGBClassifier(n_estimators=500,max_depth=6,learning_rate=.05,subsample=.9,colsample_bytree=.9,tree_method='hist',random_state=seed,n_jobs=-1,eval_metric='logloss'))])
        est.fit(X.iloc[dev],ytr); pred=est.predict(X.iloc[test]); mid=le.transform(['Malware'])[0]; bid=le.transform(['Benign'])[0]
        nheld=int(held.sum()); rows.append({'held_out_family':fam,'malware_recall':float(np.mean(pred[:nheld]==mid)),'benign_fpr':benign_fpr(yte,pred,bid),'malware_support':nheld,'benign_support':len(benign)})
    pd.DataFrame(rows).to_csv(out/'lofo_metrics.csv',index=False)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--csv',required=True,type=Path); ap.add_argument('--out',type=Path,default=Path('outputs')); ap.add_argument('--seed',type=int,default=SEED); a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    (a.out/'environment.json').write_text(json.dumps({'python':sys.version,'platform':platform.platform(),'processor':platform.processor(),'cpu_count':os.cpu_count()},indent=2),encoding='utf8')
    X,yb,yf,g=load_data(a.csv,a.out)
    run_task(X,yb,g,a.out/'binary','binary',a.seed)
    if yf is not None:
        run_task(X,yf,g,a.out/'family','family',a.seed)
        run_lofo(X,yb,yf,g,a.out/'lofo',a.seed)
    print('Finished. Inspect audit.json, split.csv, summary_metrics.csv, predictions, and confusion matrices.')
if __name__=='__main__': main()
