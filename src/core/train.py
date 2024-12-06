from utils.checkpoint import *
from utils.util import *
from tqdm import tqdm
from sklearn.metrics import  roc_auc_score, precision_recall_curve,  auc, f1_score
from collections import defaultdict
import pandas as pd
import warnings 


def eval_test(args,model,test_data_loader,device):
    model.eval()
    rootdir=args.ck_file_path

    result_file=rootdir
    os.makedirs(rootdir,exist_ok = True)

    try:
        result_dict = pickle.load(open(rootdir+"result.pkl", "rb"))
    except:
        result_dict={}

    if args.generate_data:
        seed=str(args.datagereate_seed)+'_'+str(args.seed)
    else:
        seed=args.seed
    result_dict[seed]={}
    for subdir, dirs, files in os.walk(rootdir):
        if len(files)==0 or 'pkl' in files[0]:
            continue
        substr=subdir.split('/')[-1]
        if substr=='model':
            continue

        file = str(seed) + '.pth.tar'
        file_path=os.path.join(subdir, file)
        print(file_path)
        checkpoint = torch.load(file_path)
        model.load_state_dict(checkpoint['network'])
        test_val=evaluate_irg(args=args, device=device, data_loader=test_data_loader, model=model, mode='test')
        for eval_type, val in test_val.items():
            result_dict[seed][eval_type]={}
            result_dict[seed][eval_type]['val']=checkpoint['best_val'][eval_type]
            result_dict[seed][eval_type]['test']=test_val[eval_type]

    with open(rootdir+"/result.pkl","wb") as f:
        pickle.dump(result_dict, f)


def trainer_irg(model,args,accelerator,train_dataloader,dev_dataloader,test_data_loader,device,optimizer,pretrain_epoch=None,writer=None,scheduler=None):
    count=0
    global_step=0
    best_evals={}

    # Check if the file already exists and load previous gradients
    output_file_base = '/data/wang/junh/githubs/Multimodal-Transformer/' + args.modeltype
    for epoch in tqdm(range(args.num_train_epochs)):
        cumulative_gradients = defaultdict(lambda: None)  # Initialize with None
        model.train()
        if "Text" in args.modeltype:
            if args.num_update_bert_epochs<args.num_train_epochs and (epoch)%args.num_update_bert_epochs==0 and count<args.bertcount:
                count+=1
                print("bert update at epoch "+ str(epoch) )
                for param in model.bertrep.parameters():
                        param.requires_grad = True
            else:
                for param in model.bertrep.parameters():
                    param.requires_grad = False

            for param in model.bertrep.parameters():
                print(epoch,param.requires_grad)
                break

        none_count=0
        for step, batch in tqdm(enumerate(train_dataloader)):
            if batch is None:
                none_count+=1
                continue
            global_step+=1

            ts_input_sequences, ts_mask_sequences, ts_tt, reg_ts, input_ids_sequences, attn_mask_sequences, text_emb, note_time, note_time_mask, cxr_feats, cxr_time, cxr_time_mask, ecg_feats, ecg_time, ecg_time_mask, label, cxr_missing, text_missing, ecg_missing = batch

            if  args.modeltype == "TS_Text":
                loss=model(x_ts=ts_input_sequences, \
                        x_ts_mask=ts_mask_sequences,\
                        ts_tt_list=ts_tt,\
                        input_ids_sequences=input_ids_sequences,\
                        attn_mask_sequences=attn_mask_sequences, text_emb=text_emb, note_time_list=note_time,\
                        note_time_mask_list=note_time_mask,labels=label,reg_ts=reg_ts)
            elif args.modeltype == "TS_CXR":
                loss=model(x_ts=ts_input_sequences, \
                        x_ts_mask=ts_mask_sequences,\
                        ts_tt_list=ts_tt,\
                        cxr_feats=cxr_feats,\
                        cxr_time=cxr_time, 
                        cxr_time_mask=cxr_time_mask,labels=label,reg_ts=reg_ts)
            elif args.modeltype == 'TS_CXR_Text':
                loss=model(x_ts=ts_input_sequences, \
                        x_ts_mask=ts_mask_sequences,\
                        ts_tt_list=ts_tt,\
                        input_ids_sequences=input_ids_sequences,\
                        attn_mask_sequences=attn_mask_sequences, text_emb=text_emb, note_time_list=note_time,\
                        note_time_mask_list=note_time_mask,\
                        cxr_feats=cxr_feats,\
                        cxr_time=cxr_time, \
                        cxr_time_mask=cxr_time_mask,labels=label,reg_ts=reg_ts,\
                        cxr_missing=cxr_missing, text_missing=text_missing)
            elif args.modeltype == 'TS_CXR_ECG':
                loss=model(x_ts=ts_input_sequences, \
                        x_ts_mask=ts_mask_sequences,\
                        ts_tt_list=ts_tt,\
                        input_ids_sequences=input_ids_sequences,\
                        attn_mask_sequences=attn_mask_sequences, text_emb=text_emb, note_time_list=note_time,\
                        note_time_mask_list=note_time_mask,\
                        ecg_feats=ecg_feats,\
                        ecg_time=ecg_time, \
                        ecg_time_mask=ecg_time_mask,labels=label,reg_ts=reg_ts,\
                        ecg_missing=ecg_missing, text_missing=text_missing)
            elif args.modeltype == "TS_CXR_Text_ECG":
                loss=model(x_ts=ts_input_sequences, \
                        x_ts_mask=ts_mask_sequences,\
                        ts_tt_list=ts_tt,\
                        input_ids_sequences=input_ids_sequences,\
                        attn_mask_sequences=attn_mask_sequences, text_emb=text_emb, note_time_list=note_time,\
                        note_time_mask_list=note_time_mask,\
                        cxr_feats=cxr_feats,\
                        cxr_time=cxr_time, \
                        cxr_time_mask=cxr_time_mask,\
                        ecg_feats=ecg_feats,\
                        ecg_time=ecg_time, \
                        ecg_time_mask=ecg_time_mask,labels=label,reg_ts=reg_ts,\
                        cxr_missing=cxr_missing, text_missing=text_missing, ecg_missing=ecg_missing)                
            elif args.modeltype == "TS":
                loss=model(x_ts=ts_input_sequences, \
                        x_ts_mask=ts_mask_sequences,\
                        ts_tt_list=ts_tt,\
                        labels=label,reg_ts=reg_ts)
            elif args.modeltype == "Text":
                #loss=model(input_ids_sequences=input_ids_sequences,\
                #        attn_mask_sequences=attn_mask_sequences, labels=label)
                loss=model(input_ids_sequences=input_ids_sequences,\
                        attn_mask_sequences=attn_mask_sequences, text_emb=text_emb,\
                        note_time_list=note_time, note_time_mask_list=note_time_mask,\
                        text_missing=text_missing, labels=label)
            elif args.modeltype == "CXR":
                loss=model(cxr_feats=cxr_feats,\
                        cxr_time=cxr_time, \
                        cxr_time_mask=cxr_time_mask, cxr_missing=cxr_missing, \
                        labels=label)

            if loss is None:
                # add warning
                warnings.warn("loss is None!")
                continue
            loss = loss.mean() / args.gradient_accumulation_steps
            accelerator.backward(loss)


            # Print loss and 


            if (step + 1) % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()

                # # Collect gradients by modality
                # for name, param in model.named_parameters():
                #     if param.requires_grad and "w_gate" in name:
                #         # Initialize cumulative gradient for the parameter
                #         if name not in cumulative_gradients:
                #             cumulative_gradients[name] = np.zeros_like(param.grad.detach().cpu().numpy())
                        
                #         # Add current gradient to cumulative gradient
                #         cumulative_gradients[name] += param.grad.detach().cpu().numpy()

                if scheduler is not None:
                    scheduler.step()
                model.zero_grad()

            if writer!=None:
                writer.add_scalar('training/train_loss',loss,global_step)
        # # Save cumulative gradients for this epoch
        # output_file = f"{output_file_base}_{epoch + 1}.json"  # Append epoch number
        # with open(output_file, 'w') as f:
        #     json.dump({k: v.tolist() for k, v in cumulative_gradients.items()}, f, indent=4)

        # print(f"Saved gradients for epoch {epoch + 1} to {output_file}")

        if none_count>0:
            print("none_count",none_count)

        eval_vals=evaluate_irg(args,device,dev_dataloader,model)
        for k,v in eval_vals.items():
            if k== 'auc_scores':
              continue
            if writer!=None:
                writer.add_scalar('dev/'+k ,v,epoch+1)
            best_eval=best_evals.get(k, 0)
            if v>best_eval:
                best_eval=v
                best_evals[k]=best_eval
            print("Current "+ k,v)
            print("Best "+ k,best_eval)

        if writer!=None:
            writer.close()


def evaluate_irg(args, device, data_loader, model, mode=None):
    model.eval()
    eval_logits = []
    eval_example = []
    none_count=0
    for idx, batch in enumerate(tqdm(data_loader)):
        if batch is None:
            none_count+=1
            continue
        ts_input_sequences, ts_mask_sequences, ts_tt, reg_ts, input_ids_sequences, attn_mask_sequences, text_emb, note_time, note_time_mask, cxr_feats, cxr_time, cxr_time_mask, ecg_feats, ecg_time, ecg_time_mask, label, cxr_missing, text_missing, ecg_missing = batch
        with torch.no_grad():

            if  args.modeltype == "TS_Text":
                logits=model(x_ts=ts_input_sequences, \
                        x_ts_mask=ts_mask_sequences,\
                        ts_tt_list=ts_tt,\
                        input_ids_sequences=input_ids_sequences,\
                        attn_mask_sequences=attn_mask_sequences, text_emb=text_emb, note_time_list=note_time,\
                        note_time_mask_list=note_time_mask,reg_ts=reg_ts)
            elif args.modeltype == "TS_CXR":
                logits=model(x_ts=ts_input_sequences, \
                        x_ts_mask=ts_mask_sequences,\
                        ts_tt_list=ts_tt,\
                        cxr_feats=cxr_feats,\
                        cxr_time=cxr_time, 
                        cxr_time_mask=cxr_time_mask,reg_ts=reg_ts)
            elif args.modeltype == 'TS_CXR_Text':
                logits=model(x_ts=ts_input_sequences,\
                        x_ts_mask=ts_mask_sequences,\
                        ts_tt_list=ts_tt,\
                        input_ids_sequences=input_ids_sequences,\
                        attn_mask_sequences=attn_mask_sequences, text_emb=text_emb, note_time_list=note_time,\
                        note_time_mask_list=note_time_mask,\
                        cxr_feats=cxr_feats,\
                        cxr_time=cxr_time,\
                        cxr_time_mask=cxr_time_mask, reg_ts=reg_ts,
                        cxr_missing=cxr_missing, text_missing=text_missing)
            elif args.modeltype == 'TS_CXR_Text_ECG':
                logits=model(x_ts=ts_input_sequences,\
                        x_ts_mask=ts_mask_sequences,\
                        ts_tt_list=ts_tt,\
                        input_ids_sequences=input_ids_sequences,\
                        attn_mask_sequences=attn_mask_sequences, text_emb=text_emb, note_time_list=note_time,\
                        note_time_mask_list=note_time_mask,\
                        cxr_feats=cxr_feats,\
                        cxr_time=cxr_time,\
                        cxr_time_mask=cxr_time_mask,\
                        ecg_feats=ecg_feats,\
                        ecg_time=ecg_time,\
                        ecg_time_mask=ecg_time_mask, reg_ts=reg_ts,
                        cxr_missing=cxr_missing, text_missing=text_missing, ecg_missing=ecg_missing)
            elif args.modeltype == "TS":
                logits=model(x_ts=ts_input_sequences, \
                        x_ts_mask=ts_mask_sequences,\
                        ts_tt_list=ts_tt,\
                        reg_ts=reg_ts)
            elif args.modeltype == "Text":
                #logits=model(input_ids_sequences=input_ids_sequences,\
                #        attn_mask_sequences=attn_mask_sequences)
                logits=model(input_ids_sequences=input_ids_sequences,\
                        attn_mask_sequences=attn_mask_sequences, text_emb=text_emb,\
                        note_time_list=note_time, note_time_mask_list=note_time_mask,\
                        text_missing=text_missing)
            elif args.modeltype == "CXR":
                logits=model(cxr_feats=cxr_feats,\
                        cxr_time=cxr_time, \
                        cxr_time_mask=cxr_time_mask, cxr_missing=cxr_missing)
            if logits is None:
                warnings.warn("logits is None!")
                continue
            if torch.isnan(logits).any():
                warnings.warn("logits is nan!")
                continue
            logits = logits.cpu().numpy()
            label_ids = label.cpu().numpy()
            eval_logits += logits.tolist()
            eval_example += label_ids.tolist()
        # Optional: Aggregate attention weights post-evaluation of the batch
        # Can also be done only at the end of an epoch or the complete evaluation
        # attention_weights = model.aggregate_attention_weights()
        # all_attention_weights.append(attention_weights)
        
    if none_count>0:
        print("none_count",none_count)
    
    if mode == "test":
        all_logits = np.array(eval_logits)
        all_labels = np.array(eval_example)
        predictions = np.where(all_logits > 0.5, 1, 0)

        # Check if handling a binary classification or multi-label classification
        if predictions.ndim == 1:
            # Binary classification, predictions are already 1D
            results_df = pd.DataFrame({
                "Predicted": predictions,
                "Ground_Truth": all_labels
            })
        else:
            # Multi-label classification, convert each row to tuple or string
            results_df = pd.DataFrame({
                "Predicted": [tuple(row) for row in predictions],
                "Ground_Truth": [tuple(row) for row in all_labels]
            })

        # Save to a CSV file
        output_file = f"{args.output_dir}{args.modeltype}_test_predictions.csv"
        results_df.to_csv(output_file, index=False)
        print(f"Saved test predictions to {output_file}")

    eval_vals={}
    all_logits = np.array(eval_logits)
    all_label = np.array(eval_example)
    all_pred= np.where(all_logits > 0.5, 1, 0)
    if 'pheno' in args.task:
        eval_vals=metrics_multilabel(all_label, all_logits, verbose=0)
        eval_vals['macro_f1']=f1_score(all_label, all_pred, average='macro')

        if mode==None:
            check_point(eval_vals, model, eval_logits, args,"macro_f1")

    elif 'ihm' in args.task or 'los' in args.task:
        eval_val = roc_auc_score(np.array(eval_example), np.array(eval_logits))
        eval_vals['auc']=eval_val
        (precisions, recalls, thresholds) = precision_recall_curve(np.array(eval_example), np.array(eval_logits))
        eval_val = auc(recalls, precisions)
        eval_vals['auprc']=eval_val
        eval_val=f1_score(np.array(eval_example), all_pred)
        eval_vals['f1']=eval_val
        if mode==None:
            check_point(eval_vals, model, eval_logits, args,"f1")

    return eval_vals
