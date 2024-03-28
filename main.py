import argparse
from tqdm import tqdm
import os
import json
from datetime import datetime

import numpy as np
from sklearn.metrics import mean_squared_error, classification_report, matthews_corrcoef, f1_score
from sklearn.model_selection import KFold

from dataloader import MultimodalDataset

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, get_cosine_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup


import torch
import torch.nn as nn

class MultimodalTransformer(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_heads, num_layers, dropout, max_len):
        super(MultimodalTransformer, self).__init__()

        # Define your multimodal transformer layers here
        self.video_embedding = nn.Linear(embedding_dim[0], hidden_dim)
        self.audio_embedding = nn.Linear(embedding_dim[1], hidden_dim)
        self.text_embedding = nn.Linear(embedding_dim[2], hidden_dim)

        self.transformer_layer = nn.Transformer(
            d_model=hidden_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dropout=dropout
        )

        # Add any additional layers or modifications based on your specific model architecture

    def forward(self, video, audio, text, mask, subclip_masks):
        # Apply embeddings to each modality
        video_embedded = self.video_embedding(video)
        audio_embedded = self.audio_embedding(audio)
        text_embedded = self.text_embedding(text)

        # Combine modalities (add or concatenate as needed)
        combined_input = video_embedded + audio_embedded + text_embedded

        # Pass through the transformer
        transformer_output = self.transformer_layer(combined_input, src_key_padding_mask=mask)

        # Apply any additional processing or output layers based on your specific model

        return transformer_output


def pad_collate(batch):
    target = np.array([item[0] for item in batch], dtype=np.float32)
    video = [item[1] for item in batch]
    audio = [item[2] for item in batch]
    text = [item[3] for item in batch]
    subclip_masks = [item[4] for item in batch]
    lens = [len(x) for x in video]

    video = nn.utils.rnn.pad_sequence(video, batch_first=True, padding_value=0)
    audio = nn.utils.rnn.pad_sequence(audio, batch_first=True, padding_value=0)
    text = nn.utils.rnn.pad_sequence(text, batch_first=True, padding_value=0)
    subclip_masks = nn.utils.rnn.pad_sequence(subclip_masks, batch_first=True, padding_value=0)

    lens = torch.LongTensor(lens)
    target = torch.tensor(target)

    mask = torch.arange(video.shape[1]).expand(len(lens), video.shape[1]) < lens.unsqueeze(1)
    mask = mask

    return [target, video, audio, text, mask, subclip_masks]

def positioanal_encoding(x):
    pos = np.zeros(x.shape)
    for i in tqdm(range(len(x))):
     for j in range(len(x[i])):
        for d in range(len(x[i][j])):
            if d %2 == 0:
                p = math.sin(j/pow(10000,d/62))
                pos[i][j][d] = p
            else:
                p = math.cos(j/pow(10000,(d-1)/62))
                pos[i][j][d] = p
    return pos

def train(fold, model, device, trainloader, optimizer, loss_function, epoch):
    current_loss = 0.0
    model = model.train()
    for i, data in tqdm(enumerate(trainloader, 0), total=len(trainloader)):
        targets, video, audio, text, mask, subclip_masks = data
        video = video.to(device)
        audio = audio.to(device)
        text = text.to(device)
        targets = targets.to(device)
        mask = mask.to(device)
        subclip_masks = subclip_masks.to(device)

        optimizer.zero_grad()
        outputs = model(video, audio, text, mask, subclip_masks)
        loss = loss_function(outputs, targets)
        loss.backward()
        optimizer.step()
        current_loss += loss.item()
        outputs.detach()
        del outputs

    epoch_loss = current_loss / len(trainloader)

    return epoch_loss

def test(fold, model, device, testloader, results, movement=False, val=False):
    preds = []
    true = []
    model.eval()
    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader, 0), total=len(testloader)):
            targets, video, audio, text, mask, subclip_masks = data
            video = video.to(device)
            audio = audio.to(device)
            text = text.to(device)
            targets = targets.to(device)
            mask = mask.to(device)
            subclip_masks = subclip_masks.to(device)

            outputs = model(video, audio, text, mask, subclip_masks)
            if movement:
                outputs = F.sigmoid(outputs) > 0.5
            preds.extend(outputs.detach().cpu().numpy())
            true.extend(targets.detach().cpu().numpy())

    res = {}
    preds = np.array(preds)
    true = np.array(true)
    if not movement:
        res_list = [mean_squared_error(true[:, i], preds[:, i], squared=False) for i in range(true[0].shape[-1])]
        res['rmse'] = res_list
    else:
        res_list = [f1_score(true[:, i], preds[:, i]) for i in range(true[0].shape[-1])]
        res['f1-score'] = res_list
        res_list = [matthews_corrcoef(true[:, i], preds[:, i]) for i in range(true[0].shape[-1])]
        res['mcc'] = res_list

    results[fold] = res

    return results


def main(config):
    EPOCHS = config.epochs
    BATCH_SIZE = config.batch_size
    HIDDEN_DIM = config.hidden_dim
    EMBEDDING_DIM = [int(e) for e in config.embedding_dim]
    DROPOUT = config.dropout
    DATA_DIR = config.data_dir
    NUM_LAYERS = config.num_layers
    NUM_HEADS = config.num_heads
    N_FOLDS = config.n_folds
    MAX_LEN = config.max_len
    LEARNING_RATE = config.learning_rate
    MODEL = config.model
    SAVE_DIR = config.save_dir
    OPTIMIZER = config.optimizer
    SCHEDULER = config.use_scheduler
    PATIENCE = config.patience
    MIN_EPOCHS = config.min_epochs
    MOVEMENT = config.movement
    OFFSET = config.offset
    SUBCLIP_MAXLEN = config.subclip_maxlen
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device: ", DEVICE)

    dataset = MultimodalDataset(DATA_DIR, SUBCLIP_MAXLEN)
    dataset.load_data(DATA_DIR, OFFSET, MOVEMENT)
    train_idx, val_idx, test_idx = dataset.make_splits()

    kfold = KFold(n_splits=N_FOLDS, shuffle=True)
    results_val = {}
    results = {}

    if MOVEMENT:
        loss_function = nn.BCEWithLogitsLoss()
    else:
        loss_function = nn.MSELoss()

    #for fold,(train_idx,test_idx) in enumerate(kfold.split(dataset.timestamps)):
    for fold in range(N_FOLDS):
        print('------------fold no---------{}----------------------'.format(fold))

        train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_idx)
    
        trainloader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=train_subsampler, collate_fn=pad_collate)
        valloader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=val_subsampler, collate_fn=pad_collate)
        testloader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=test_subsampler, collate_fn=pad_collate)
        ##import model
        model = Model(EMBEDDING_DIM, HIDDEN_DIM, NUM_HEADS, NUM_LAYERS, DROPOUT, MAX_LEN)
        model.train()
        model.to(DEVICE)

        if OPTIMIZER=="adamw":
            optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
        else:  
            optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        if SCHEDULER:
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps=5, num_training_steps=EPOCHS
            )
        else:
            scheduler = None
          
        best_metric = np.inf
        counter = 0
        for epoch in range(1, EPOCHS + 1):
            current_loss = train(fold, model, DEVICE, trainloader, optimizer, loss_function, epoch)
            results_val = test(fold, model, DEVICE, valloader, results_val, val=True, movement=MOVEMENT)
            
            if MOVEMENT:
                print('Epoch %5d | Avg Train Loss %.3f | Val F1s  %s'% (epoch, current_loss, str(results_val[fold]['f1-score'])))
            else:
                print('Epoch %5d | Avg Train RMSE %.3f | Val RMSEs %s'% (epoch, np.sqrt(current_loss), str(results_val[fold]['rmse'])))

            if scheduler is not None:
                scheduler.step()
            if current_loss < best_metric:
                counter=0
            else:
                counter += 1
                if counter > PATIENCE and epoch > MIN_EPOCHS:
                    print("Early stopping")
                    break
        
        results = test(fold, model, DEVICE, testloader, results)
        

    print(f'K-FOLD CROSS VALIDATION RESULTS FOR {N_FOLDS} FOLDS')
    print('--------------------------------')
    sums = []
    mccs = []
    for key, value in results.items():
        #print(f'Fold {key}: {value} %')
        if MOVEMENT:
            sums.append(np.array(value['report']['macro avg']['f1-score']))
            mccs.append(np.array(value['mcc']))
        else:
            sums.append(np.array(value['rmse']))
    avgs = np.mean(sums, axis=0)
    print(f'Average over N runs F1/RMSE: {avgs} %')
    if MOVEMENT:
        avgs = np.mean(mccs, axis=0)
        print(f'Average over N runs MCC: {avgs} %')

    save_dir = SAVE_DIR
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    save_folder = os.path.join(save_dir, datetime.today().strftime('%Y-%m-%d'))
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    save_dict = vars(config)
    save_dict["results"] = results

    now = datetime.now()
    current_time = now.strftime("%H-%M-%S")

    keys_values = save_dict.items()
    save_dict = {str(key): str(value) for key, value in keys_values}
    filename = os.path.join(save_folder, current_time) + ".json"
    with open(filename, "w") as outfile:
        json.dump(save_dict, outfile, indent=2)

    return avgs
    


if __name__ == '__main__':
    model_set = {"mult"}
    optimizer_set = {"adam", "adamw"}

    parser = argparse.ArgumentParser(description="Multimodal Transformer")
    parser.add_argument("-lr", "--learning-rate", default=1e-3, type=float)
    parser.add_argument("-bs", "--batch-size", default=1, type=int)
    parser.add_argument("-e", "--epochs", default=30, type=int)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--min-epochs', type=int, default=10)
    parser.add_argument("-hd", "--hidden-dim", default=16, type=int)
    parser.add_argument('-ed','--embedding-dim', nargs='+', default=['768', '768', '768'])
    parser.add_argument("-nl", "--num-layers", default=2, type=int)
    parser.add_argument("-nh", "--num-heads", default=4, type=int)
    parser.add_argument("-ml", "--max-len", default=2048, type=int)
    parser.add_argument("-d", "--dropout", default=0.3, type=float)
    parser.add_argument("-nf", "--n-folds", default=3, type=int)
    parser.add_argument("-sm", "--subclip-maxlen", default=-1, type=int)
    parser.add_argument("--model", type=str, choices=model_set, default="mult")
    parser.add_argument("--optimizer", type=str, choices=optimizer_set, default="adamw")
    parser.add_argument("--use-scheduler", action="store_true")
    parser.add_argument("--movement", action="store_true")
    parser.add_argument("-o", "--offset", default=1, type=int)
    parser.add_argument("--data-dir", type=str, default="/home/ouyangkun/MPC/data")
    parser.add_argument("--save-dir", type=str, default="../../results")
    config = parser.parse_args()

    main(config)