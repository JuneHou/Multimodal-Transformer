from utils.checkpoint import *
from utils.util import *
from tqdm import tqdm
from sklearn.metrics import  roc_auc_score, precision_recall_curve,  auc, f1_score, confusion_matrix, classification_report
from collections import defaultdict
import pandas as pd
import warnings 


def eval_test(args, model, dataloader, device, mode=None):
    print("Mode: ", mode)
    model.eval()  # Set model to evaluation mode.

    # Get the directory for storing results using the construct_path function
    rootdir = construct_path(args, 'f1')  # Assuming 'f1' is used for evaluation metrics categorization
    os.makedirs(rootdir, exist_ok=True)

    # Load the result dictionary if exists
    result_dict_path = os.path.join(rootdir, "result.pkl")
    try:
        result_dict = pickle.load(open(result_dict_path, "rb"))
    except FileNotFoundError:
        result_dict = {}

    seed = str(args.seed)
    result_dict[seed] = {}

    # Load the best model checkpoint for evaluation if not in testing mode
    if mode in ["train", "val"]:
        best_model_file = find_best_model_file(rootdir)
        if best_model_file:
            print("Loading best model from:", best_model_file)
            checkpoint = torch.load(best_model_file, map_location=device)
            model.load_state_dict(checkpoint['network'])
        else:
            print("No best model found. Evaluating with the current model state.")

    # Perform evaluation
    eval_vals = evaluate_irg(args=args, device=device, data_loader=dataloader, model=model, mode=mode)
    for eval_type, val in eval_vals.items():
        result_dict[seed][eval_type] = {}
        result_dict[seed][eval_type][mode] = val
        if mode in ["train", "val"]:
            result_dict[seed][eval_type]['best_val'] = checkpoint['best_val'][eval_type] if best_model_file else None

    # Save results to a pickle file if in test mode
    if mode == "test":
        with open(result_dict_path, "wb") as f:
            pickle.dump(result_dict, f)

    return result_dict

def find_best_model_file(directory):
    """Utility to find the best model file based on validation performance."""
    best_model_file = None
    best_val = float('-inf')
    for file in os.listdir(directory):
        if file.endswith('.pth.tar'):
            file_path = os.path.join(directory, file)
            try:
                checkpoint = torch.load(file_path)
                if 'best_val' in checkpoint and checkpoint['best_val']['val'] > best_val:
                    best_val = checkpoint['best_val']['val']
                    best_model_file = file_path
            except Exception as e:
                print("Error loading checkpoint:", e)
    return best_model_file

def batch_input_fields(batch, model_type, train=True):

    keys = ['ids', 'x_ts', 'x_ts_mask', 'ts_tt_list', 'reg_ts', 'input_ids_sequences', \
            'attn_mask_sequences', 'text_emb', 'note_time_list', 'note_time_mask_list', 'cxr_feats', 'cxr_time', \
            'cxr_time_mask', 'ecg_feats', 'ecg_time', 'ecg_time_mask', 'labels', 'cxr_missing', 'text_missing', \
            'ecg_missing', 'ts_weight', 'text_weight', 'cxr_weight', 'ecg_weight']
    batch_fields = {key: batch[i] for i, key in enumerate(keys)}


    model_input_fields = {'TS': ['x_ts', 'x_ts_mask', 'ts_tt_list', 'reg_ts'],
                          'Text': ['input_ids_sequences', 'attn_mask_sequences', 'text_emb',
                                'note_time_list', 'note_time_mask_list', 'text_missing'],
                          'CXR': ['cxr_feats', 'cxr_time', 'cxr_time_mask', 'cxr_missing'],
                          'ECG': ['ecg_feats', 'ecg_time', 'ecg_time_mask', 'ecg_missing']}


    input_fields = {'ts_weight': batch_fields['ts_weight'], 'text_weight': batch_fields['text_weight'],
                'cxr_weight': batch_fields['cxr_weight'], 'ecg_weight': batch_fields['ecg_weight']}

    if train:
        input_fields['labels'] = batch_fields['labels']

    for key in model_input_fields.keys():
        if key in model_type:
            for field in model_input_fields[key]:
                input_fields[field] = batch_fields[field]

    return input_fields, batch_fields
        


def trainer_irg(model,args,accelerator,train_dataloader,dev_dataloader,test_data_loader,device,optimizer,pretrain_epoch=None,writer=None,scheduler=None):
    count=0
    global_step=0
    best_evals={}

    # Check if the file already exists and load previous gradients
    output_file_base = '/data/wang/junh/githubs/Multimodal-Transformer/' + args.modeltype
    for epoch in tqdm(range(args.num_train_epochs)):
        cumulative_gradients = defaultdict(lambda: None)  # Initialize with Non
        all_gradients = []
        all_ids = []
        
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

            input_fields, _ = batch_input_fields(batch, args.modeltype)

            loss, probs = model(**input_fields)


            if loss is None:
                # add warning
                warnings.warn("loss is None!")
                continue
            loss = loss.mean() / args.gradient_accumulation_steps
            accelerator.backward(loss)
            
            # if epoch == args.num_train_epochs - 1:  # Collect data only in the last epoch
            #     batch_gradients = {}
            #     for name, param in model.named_parameters():
            #         if param.requires_grad and "w_gate" in name:
            #             batch_gradients[name] = [param.grad[i].detach().cpu().numpy().tolist() for i in range(args.train_batch_size)]

            #     # Collect ids and gradients for each instance in the batch
            #     all_ids.extend(ids)
            #     all_gradients.append(batch_gradients)

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
        
        # if epoch == args.num_train_epochs - 1:  # Save at the end of the last epoch
        #     # Option 2: Save to CSV file (if feasible)
        #     # Flattening gradients for CSV is more complex and depends on your specific needs
        #     # This example assumes you can serialize gradients as strings or handle them appropriately
        #     gradient_data = {
        #         "id": all_ids,
        #         "gradients": [json.dumps({k: v for k, v in batch.items()}) for batch in all_gradients]
        #     }
        #     df = pd.DataFrame(gradient_data)
        #     csv_file = f"{args.output_dir}/{args.modeltype}_gradients_epoch_{epoch + 1}.csv"
        #     df.to_csv(csv_file, index=False)

        #     print(f"Saved gradients to {csv_file}")

        # # Save cumulative gradients, probs, and predictions for this epoch
        # gradient_file = f"{args.output_dir}{args.modeltype}_gradients_{epoch + 1}.json"
        # probs_file = f"{args.output_dir}{args.modeltype}_probs_{epoch + 1}.json"

        # with open(gradient_file, 'w') as f:
        #     json.dump({k: v.tolist() for k, v in cumulative_gradients.items()}, f, indent=4)

        # print(f"Saved gradients to {gradient_file}")

        if none_count>0:
            print("none_count",none_count)

        eval_vals=evaluate_irg(args,device,dev_dataloader,model)
        print(eval_vals)
        # for k,v in eval_vals.items():
        #     if k== 'auc_scores':
        #         continue
        #     if writer!=None:
        #         writer.add_scalar('dev/'+k ,v,epoch+1)
        #     best_eval=best_evals.get(k, 0)
        #     if v>best_eval:
        #         best_eval=v
        #         best_evals[k]=best_eval
        #     print("Current "+ k,v)
        #     print("Best "+ k,best_eval)

        if writer!=None:
            writer.close()


def evaluate_irg(args, device, data_loader, model, mode=None):
    model.eval()

    eval_ids = []
    eval_proj = []
    eval_probs = []
    record_probs = []
    eval_labels = []
    none_count=0

    for idx, batch in enumerate(tqdm(data_loader)):
        if batch is None:
            none_count+=1
            continue
        
        with torch.no_grad():

            input_fields, batch = batch_input_fields(batch, args.modeltype, train=False)

            label = batch['labels']
            ids = batch['ids']

            proj, probs = model(**input_fields)

            if probs is None:
                warnings.warn("probs is None!")
                continue
            if torch.isnan(probs).any():
                warnings.warn("probs is nan!")
                continue
            probs = probs.cpu().numpy()
            label = label.cpu().numpy()
            proj = proj.cpu().numpy()
            eval_proj += proj.tolist()
            eval_probs += probs.tolist()
            record_probs.extend(probs.tolist())
            eval_labels += label.tolist()
            eval_ids += ids
        # Optional: Aggregate attention weights post-evaluation of the batch
        # Can also be done only at the end of an epoch or the complete evaluation
        # attention_weights = model.aggregate_attention_weights()
        # all_attention_weights.append(attention_weights)
        
    if none_count>0:
        print("none_count",none_count)
    
    if mode is not None:
        all_probs = np.array(record_probs)
        all_proj = np.stack(eval_proj)
        all_labels = np.array(eval_labels)
        all_ids = np.array(eval_ids)
        
        # Check if task is binary or multiclass based on the shape of all_probs
        if all_probs.shape[1] == 1 or len(np.unique(all_labels)) == 2:
            # Binary classification task
            predictions = np.where(all_probs > 0.5, 1, 0).flatten()  # Convert to 0 or 1 and flatten array
        else:
            # Multiclass classification task
            predictions = np.argmax(all_probs, axis=1)
            all_probs = list(map(list, all_probs))

        # Check if handling a binary classification or multi-label classification
        if predictions.ndim == 1 or all_labels.shape[1] == 1:
            # Binary classification, predictions are already 1D
            results_df = pd.DataFrame({
                "ids": all_ids,
                "Predicted": predictions,
                "Ground_Truth": all_labels,
                "Probs": all_probs,
                "Proj": list(map(list, all_proj))
            })
        else:
            # Multi-label classification, convert each row to tuple or string
            results_df = pd.DataFrame({
                "ids": all_ids,
                "Predicted": [tuple(row) for row in predictions],
                "Ground_Truth": [tuple(row) for row in all_labels],
                "Probs": all_probs,
                "Proj": list(map(list, all_proj))
            })

        # Save to a CSV file
        #output_file = f"{args.output_dir}/{args.task}_{args.modeltype}_{mode}_results.csv"
        output_file = f"{args.output_dir}/{args.modeltype}_{mode}_results.csv"
        results_df.to_csv(output_file, index=False)
        print(f"Saved test predictions to {output_file}")
    
    eval_vals={}
    all_probs = np.array(eval_probs)
    all_label = np.array(eval_labels)
    all_pred= np.where(all_probs > 0.5, 1, 0)
    if 'pheno' in args.task:
        eval_vals=metrics_multilabel(all_label, all_probs, verbose=0)
        eval_vals['macro_f1']=f1_score(all_label, all_pred, average='macro')

        if mode==None:
            check_point(eval_vals, model, eval_probs, args,"macro_f1")

    if 'ihm' in args.task:
        # Continue using binary classification metrics for IHM
        eval_val = roc_auc_score(all_label, all_probs)
        eval_vals['auc'] = eval_val
        (precisions, recalls, thresholds) = precision_recall_curve(all_label, all_probs)
        eval_val = auc(recalls, precisions)
        eval_vals['auprc'] = eval_val
        eval_val = f1_score(all_label, all_pred, average='binary')
        eval_vals['f1'] = eval_val
        if mode is None:
            check_point(eval_vals, model, eval_probs, args, "f1")

    elif 'los' in args.task:
        # Convert probs to class predictions for multiclass classification.
        all_pred = np.argmax(all_probs, axis=1)  # Use argmax to get class with highest probability.

        # Use multiclass classification metrics for LOS.
        eval_vals['confusion_matrix'] = confusion_matrix(all_label, all_pred)
        eval_vals['classification_report'] = classification_report(all_label, all_pred, target_names=['≤ 3 days', '> 3 & ≤ 7 days', '> 7 & ≤ 14 days', '> 14 days'])
        #eval_vals['classification_report'] = classification_report(all_label, all_pred, target_names=['≤ 3 days', '> 3 & ≤ 7 days', '> 7 & ≤ 14 days'])
        eval_val = f1_score(all_label, all_pred, average='macro')
        eval_vals['f1'] = eval_val
        if mode is None:
            check_point(eval_vals, model, eval_probs, args, "f1")

    # elif 'ihm' in args.task or 'los' in args.task:
    #     eval_val = roc_auc_score(np.array(eval_labels), np.array(eval_probs))
    #     eval_vals['auc']=eval_val
    #     (precisions, recalls, thresholds) = precision_recall_curve(np.array(eval_labels), np.array(eval_probs))
    #     eval_val = auc(recalls, precisions)
    #     eval_vals['auprc']=eval_val
    #     eval_val=f1_score(np.array(eval_labels), all_pred)
    #     eval_vals['f1']=eval_val
    #     if mode==None:
    #         check_point(eval_vals, model, eval_probs, args,"f1")

    return eval_vals
