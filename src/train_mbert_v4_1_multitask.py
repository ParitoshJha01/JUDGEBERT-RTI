"""v4 multi-task mBERT for RTI Legal Meaning Preservation.
Tasks: LMP score regression (1-10) + Critical_Change binary classification (0/1).
Run after manually reviewing Critical_Change labels in the v4 annotation CSV.

v4.1: Trains on the combined 512-row CSV.
Expansion examples are TRAIN ONLY; original validation/test remain unchanged.

From project root:
  python src/train_mbert_v4_1_multitask.py
"""
from pathlib import Path
import argparse, json, random
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.metrics import mean_absolute_error, mean_squared_error, f1_score, precision_score, recall_score, accuracy_score
from scipy.stats import pearsonr, spearmanr

MODEL_NAME="google-bert/bert-base-multilingual-cased"
MAX_LENGTH=256
BATCH_SIZE=8
EPOCHS=20
LR=2e-5
WEIGHT_DECAY=0.01
PATIENCE=4
SEED=123
CLASS_LOSS_WEIGHT=0.5

class PairDataset(Dataset):
    def __init__(self, frame, tokenizer):
        self.frame=frame.reset_index(drop=True)
        self.tokenizer=tokenizer
    def __len__(self): return len(self.frame)
    def __getitem__(self, i):
        r=self.frame.iloc[i]
        enc=self.tokenizer(str(r.Original_Text),str(r.Simplified_Text),truncation=True,max_length=MAX_LENGTH,
                          padding="max_length",return_tensors="pt")
        item={k:v.squeeze(0) for k,v in enc.items()}
        item["lmp_label"]=torch.tensor(float(r.LMP_Score),dtype=torch.float32)
        cc=r.get("Critical_Change",np.nan)
        try: cc=float(cc)
        except (TypeError,ValueError): cc=float("nan")
        item["cc_label"]=torch.tensor(cc,dtype=torch.float32)
        return item

class MultiTaskModel(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder=encoder
        h=encoder.config.hidden_size
        self.dropout=nn.Dropout(0.1)
        self.regression_head=nn.Linear(h,1)
        self.critical_head=nn.Linear(h,1)
    def forward(self,input_ids,attention_mask,token_type_ids=None):
        out=self.encoder(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
        cls= self.dropout(out.last_hidden_state[:,0,:])
        return self.regression_head(cls).squeeze(-1), self.critical_head(cls).squeeze(-1)

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def metric_reg(y,p):
    if len(y)==0: return {"N":0,"MAE":None,"RMSE":None,"Pearson_r":None,"Spearman_rho":None}
    pear=float(pearsonr(y,p).statistic) if len(y)>1 and np.std(y)>0 and np.std(p)>0 else None
    spear=float(spearmanr(y,p).statistic) if len(y)>1 and np.std(y)>0 and np.std(p)>0 else None
    return {"N":int(len(y)),"MAE":float(mean_absolute_error(y,p)),"RMSE":float(np.sqrt(mean_squared_error(y,p))),"Pearson_r":pear,"Spearman_rho":spear}

def evaluate(model,loader,device):
    model.eval(); ys=[]; ps=[]; cys=[]; cps=[]
    with torch.no_grad():
        for b in loader:
            ids=b["input_ids"].to(device); mask=b["attention_mask"].to(device)
            tt=b.get("token_type_ids"); tt=tt.to(device) if tt is not None else None
            pr,cl=model(ids,mask,tt)
            ys.extend(b["lmp_label"].numpy()); ps.extend(pr.cpu().numpy())
            cc=b["cc_label"].numpy(); valid=np.isfinite(cc)
            if valid.any():
                cys.extend(cc[valid]); cps.extend(torch.sigmoid(cl).cpu().numpy()[valid])
    p=np.clip(np.asarray(ps),1,10); y=np.asarray(ys)
    reg=metric_reg(y,p)
    if len(cys):
        pred=(np.asarray(cps)>=0.5).astype(int); true=np.asarray(cys).astype(int)
        cls={"N":len(true),"Accuracy":float(accuracy_score(true,pred)),"Precision":float(precision_score(true,pred,zero_division=0)),
             "Recall":float(recall_score(true,pred,zero_division=0)),"F1":float(f1_score(true,pred,zero_division=0))}
    else: cls={"N":0,"Accuracy":None,"Precision":None,"Recall":None,"F1":None}
    return reg,cls,y,p,np.asarray(cys),np.asarray(cps)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',default='dataset/JUDGEBERT_v4_1_Combined_512_Grouped.csv')
    ap.add_argument('--output',default='output/mbert_v4_1_multitask')
    ap.add_argument('--class-loss-weight',type=float,default=CLASS_LOSS_WEIGHT)
    args=ap.parse_args()
    data=Path(args.data); out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    if not data.exists(): raise FileNotFoundError(f"Dataset not found: {data}")
    df=pd.read_csv(data)
    # v4.1 guardrails: keep expansion data in train and preserve original held-out sets.
    if 'Dataset_Source' in df.columns:
        expansion=df['Dataset_Source'].astype(str).str.contains('expansion',case=False,na=False)
        bad= df.loc[expansion & ~df['Split'].astype(str).str.lower().eq('train')]
        if len(bad):
            raise ValueError(f'{len(bad)} expansion rows are not assigned to train; stop to prevent leakage.')
    print(f'Loaded {len(df)} rows from {data.resolve()}')
    if 'Dataset_Source' in df.columns:
        print('Rows by source:')
        print(df['Dataset_Source'].fillna('unknown').value_counts().to_string())
    print('Rows by split:')
    print(df['Split'].astype(str).str.lower().value_counts().to_string())
    required=['Pair_ID','Language','Original_Text','Simplified_Text','LMP_Score','Split','Critical_Change']
    missing=[c for c in required if c not in df.columns]
    if missing: raise ValueError(f"Missing columns: {missing}")
    df['LMP_Score']=pd.to_numeric(df['LMP_Score'],errors='raise')
    df['Critical_Change']=pd.to_numeric(df['Critical_Change'],errors='coerce')
    if not df['LMP_Score'].between(1,10).all(): raise ValueError('LMP_Score must be 1-10.')
    known=df['Critical_Change'].dropna()
    if not set(known.unique()).issubset({0,1}): raise ValueError('Critical_Change must be blank, 0, or 1.')
    # Ignore unresolved annotation rows for classification, while retaining them for LMP regression.
    if 'Annotation_Status' in df.columns:
        unresolved=df['Annotation_Status'].astype(str).str.strip().str.lower().eq('needs review')
        df.loc[unresolved,'Critical_Change']=np.nan
    print('Critical-change labels by split (known 0/1 only):')
    for split_name in ['train','validation','test']:
        part=df[df['Split'].astype(str).str.strip().str.lower().eq(split_name)]
        labels=part['Critical_Change'].dropna().astype(int)
        unresolved_count=int(part['Critical_Change'].isna().sum())
        print(f"  {split_name}: known={len(labels)}, class_0={(labels==0).sum()}, class_1={(labels==1).sum()}, unresolved={unresolved_count}")
    tr=df[df.Split.eq('train')].copy(); va=df[df.Split.eq('validation')].copy(); te=df[df.Split.eq('test')].copy()
    if min(len(tr),len(va),len(te))==0: raise ValueError('Train/validation/test splits must all exist.')
    tr_labeled=tr['Critical_Change'].notna()
    if tr_labeled.sum()<20: raise ValueError(f'Only {tr_labeled.sum()} labeled training rows. Annotate at least 20; preferably all train rows.')
    classes=set(tr.loc[tr_labeled,'Critical_Change'].astype(int).unique())
    if classes!={0,1}: raise ValueError('Training labels must contain both Critical_Change classes 0 and 1.')
    seed_all(SEED)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:',device, 'GPU:',torch.cuda.get_device_name(0) if device.type=='cuda' else 'CPU')
    tokenizer=AutoTokenizer.from_pretrained(MODEL_NAME)
    train_loader=DataLoader(PairDataset(tr,tokenizer),batch_size=BATCH_SIZE,shuffle=True)
    val_loader=DataLoader(PairDataset(va,tokenizer),batch_size=BATCH_SIZE,shuffle=False)
    test_loader=DataLoader(PairDataset(te,tokenizer),batch_size=BATCH_SIZE,shuffle=False)
    model=MultiTaskModel(AutoModel.from_pretrained(MODEL_NAME)).to(device)
    n0=(tr.loc[tr_labeled,'Critical_Change']==0).sum(); n1=(tr.loc[tr_labeled,'Critical_Change']==1).sum()
    pos_weight=torch.tensor([n0/max(n1,1)],dtype=torch.float32,device=device)
    reg_loss=nn.HuberLoss(delta=1.0); cls_loss=nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)
    total_steps=len(train_loader)*EPOCHS
    sched=get_linear_schedule_with_warmup(opt,num_warmup_steps=max(1,int(.1*total_steps)),num_training_steps=total_steps)
    best=float('inf'); best_epoch=0; stale=0; hist=[]; ckpt=out/'best_v4_1_multitask.pt'
    for epoch in range(1,EPOCHS+1):
        model.train(); losses=[]
        for b in train_loader:
            ids=b['input_ids'].to(device); mask=b['attention_mask'].to(device)
            tt=b.get('token_type_ids'); tt=tt.to(device) if tt is not None else None
            y=b['lmp_label'].to(device); cc=b['cc_label'].to(device)
            opt.zero_grad(set_to_none=True); pred,logits=model(ids,mask,tt)
            loss_r=reg_loss(pred,y)
            valid=torch.isfinite(cc)
            if valid.any(): loss_c=cls_loss(logits[valid],cc[valid])
            else: loss_c=torch.zeros((),device=device)
            loss=loss_r+args.class_loss_weight*loss_c
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); sched.step()
            losses.append(float(loss.detach().cpu()))
        val_reg,val_cls,*_=evaluate(model,val_loader,device)
        row={'epoch':epoch,'train_loss':float(np.mean(losses)),**{'val_'+k:v for k,v in val_reg.items()},**{'val_cc_'+k:v for k,v in val_cls.items()}}
        hist.append(row); print(row)
        score=val_reg['RMSE']
        if score is not None and score<best:
            best=score; best_epoch=epoch; stale=0
            torch.save({'model_state_dict':model.state_dict(),'model_name':MODEL_NAME,'max_length':MAX_LENGTH,
                       'best_epoch':best_epoch,'best_validation_rmse':best,'class_loss_weight':args.class_loss_weight,
                       'critical_change_threshold':0.5},ckpt)
            model.encoder.save_pretrained(out/'encoder'); tokenizer.save_pretrained(out/'encoder')
        else:
            stale+=1
            if stale>=PATIENCE: print('Early stopping.'); break
        pd.DataFrame(hist).to_csv(out/'training_history.csv',index=False,encoding='utf-8-sig')
    saved=torch.load(ckpt,map_location=device,weights_only=False); model.load_state_dict(saved['model_state_dict'])
    reg,cls,y,p,cy,cp=evaluate(model,test_loader,device)
    test_result={'best_epoch':best_epoch,'best_validation_rmse':best,'test_regression':reg,'test_critical_change':cls,
                 'critical_change_threshold':0.5,'note':'v4.1 experimental run. Synthetic expansion rows are train-only; original validation/test remain unchanged. Labels require human review/adjudication.'}
    (out/'test_metrics.json').write_text(json.dumps(test_result,indent=2),encoding='utf-8')
    # preserve row-level prediction outputs
    model.eval(); rows=[]; offset=0
    with torch.no_grad():
        for b in test_loader:
            ids=b['input_ids'].to(device); mask=b['attention_mask'].to(device); tt=b.get('token_type_ids'); tt=tt.to(device) if tt is not None else None
            pr,cl=model(ids,mask,tt); probs=torch.sigmoid(cl).cpu().numpy(); pr=np.clip(pr.cpu().numpy(),1,10)
            cc=b['cc_label'].numpy()
            for j in range(len(pr)):
                rows.append({'test_index':offset,'LMP_Score':float(b['lmp_label'][j]),'Predicted_LMP_Score':float(pr[j]),
                             'Critical_Change':None if not np.isfinite(cc[j]) else int(cc[j]),
                             'Critical_Change_Probability':float(probs[j]),
                             'Predicted_Critical_Change':int(probs[j]>=0.5)})
                offset+=1
    pred_df=te[['Pair_ID','Language','Split','Source_Group','LMP_Score','Critical_Change']].reset_index(drop=True).copy()
    extra=pd.DataFrame(rows).drop(columns=['test_index'])
    pred_df['Predicted_LMP_Score']=extra['Predicted_LMP_Score']; pred_df['Critical_Change_Probability']=extra['Critical_Change_Probability']; pred_df['Predicted_Critical_Change']=extra['Predicted_Critical_Change']
    pred_df.to_csv(out/'test_predictions.csv',index=False,encoding='utf-8-sig')
    print('\nTest metrics:',json.dumps(test_result,indent=2)); print('Saved to',out.resolve())

if __name__=='__main__': main()
