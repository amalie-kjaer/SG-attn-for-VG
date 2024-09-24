import numpy as np
import wandb
from tqdm import tqdm
import torch.optim
import torch.nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from pyg_dataset import ScanNetPyG
from VG_models.CLIP_GNN.model import VQAModel_attn, CLIP_GNN, Baseline, visualize_attention
from loss import FocalLoss
import sys
sys.path.append('..')
from utils import load_json, load_config, set_seed, wandb_logging, save_ckpt, write_json

# TODO add resume training

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
set_seed(1000)

def load_dataset(config,
    type: str,
    shuffle=True,
    verbose=True):

    assert type in ['train', 'test'], "Select type from ['train', 'test']."
    if verbose: print(f'Loading {type}ing dataset...')
    split_file_type = f'split_file_{type}'
    
    dataset = ScanNetPyG(
        root=config['paths']['scannet_data_root'],
        split_file_path=config['paths'][split_file_type],
        cropping_method=config['model']['cropping_method'],
        qa_dataset=config['paths']['scanqa_data']
    )
    dataloader = DataLoader(
        dataset,
        batch_size=int(config['training']['batch_size']),
        shuffle=shuffle,
        drop_last=False
    )
    return dataset, dataloader

def load_model(config):
    model_name = config['model']['model_name']
    assert model_name in ['VQAModel_attn', 'CLIP_GNN', 'Baseline'], "Select model from ['VQAModel_attn', 'CLIP_GNN', 'Baseline']"
    print(f'Loading model {model_name}...')
    
    if model_name == 'CLIP_GNN':
        model = CLIP_GNN(
            max_seq_length=35,
            gnn_type='GCN',
        )
    
    elif model_name == 'VQAModel_attn':
        model = VQAModel_attn(
            # GCN Scene graph encoding
            gnn_type=str(config['model']['gnn_type']),
            # Question encoding
            embed_dim=512,
            num_heads=int(config['model']['num_heads']),
            num_encoder_layers=int(config['model']['num_encoder_layers']),
            ff_dim=int(config['model']['ff_dim']),
            max_seq_length=int(config['model']['max_seq_length']),
            # Cross attention
            xattn_num_layers=int(config['model']['xattn_num_layers']),
            xattn_ff_dim=int(config['model']['xattn_ff_dim']),
            xattn_num_heads=int(config['model']['xattn_num_heads']),
            dropout=float(config['model']['dropout']),
            ff_dim_downsize=int(config['model']['ff_dim_downsize'])
        )
    
    elif model_name =='Baseline':
        model = Baseline(max_seq_length=35)
    
    return model.to(device)

def train_model(config_path):
    print('Loading config...')
    config = load_config(config_path)
    wandb_logging(config=config, mode='init')
    
    # -------------------
    # Load datasets and vocabulary
    # -------------------
    train_dataset, train_loader = load_dataset(config, 'train', shuffle=True)
    test_dataset, test_loader = load_dataset(config, 'test', shuffle=False)
    vocab = load_json('vocab_all_scenes.json')

    # -------------------
    # Initialize model
    # -------------------
    model = load_model(config)
    # print(model.device, flush=True) # TODO check model device
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config['training']['lr']),
        weight_decay=float(config['training']['weight_decay'])   
    )
    # criterion = torch.nn.CrossEntropyLoss()
    criterion = FocalLoss(gamma=2.0)

    # -------------------
    # Train model
    # -------------------
    for e in range(int(config['training']['epochs'])):
        print(f"Epoch {e+1}\n-------------------------------")
        
        loss = train_epoch(model, train_loader, optimizer, criterion)
        train_acc, train_loss, _, _, _, _, _, _ = test_epoch(model, train_loader, criterion, vocab)
        test_acc, test_loss, _, em5, em10, test_sem_acc, _, _ = test_epoch(model, test_loader, criterion, vocab)

        wandb_logging(config=config, mode='log', epoch=e,
            data={
                'Criterion': loss,
                'Train accuracy': train_acc,
                'Train loss': train_loss,
                'Test accuracy': test_acc,
                'Test loss': test_loss,
                'EM@5': em5,
                'EM@10': em10,
                'Semantic test accuracy': test_sem_acc
            }
        )
        save_ckpt(config=config, epoch=e, model=model)

def train_epoch(model, train_loader, optimizer, criterion, verbose=True):
    print('Training...')
    
    model.train()
    for graphs, questions, answer_instances, _, scene_id in tqdm(train_loader):
        graphs = graphs.to(device)
        # print(graphs.device, flush=True) # TODO check data device
        answer_instances = answer_instances.to(device) #TODO

        optimizer.zero_grad()
        logits, _, _ = model(graphs, questions)

        loss = criterion(logits.squeeze(-1), answer_instances)
        loss.backward()
        optimizer.step()
    
    if verbose: print(f'Epoch loss: {loss}')
    return loss

def test_epoch(model, test_loader, criterion, vocab, verbose=False):
    print('Testing...')

    model.eval()
    size = len(test_loader.dataset)
    num_batches = len(test_loader)
    test_loss, correct, em5, em10, semantic_correct = 0, 0, 0, 0, 0
    with torch.no_grad():
        for graphs, questions, answer_instances, answer_labels, scene_id in tqdm(test_loader):
            graphs = graphs.to(device)
            answer_instances = answer_instances.to(device)
            # print(graphs.device, flush=True) # TODO check data device
            # TODO: Use answer_labels to calculate semantic accuracy

            logits, attn_weights, word_tokens = model(graphs, questions) # [batch_size, max_node_length, 1]
            pred = torch.argmax(logits.squeeze(-1), dim=1) # [batch_size] 
            top5 = torch.topk(logits, 5, dim=1).indices
            top10 = torch.topk(logits, 10, dim=1).indices
            
            if verbose: print(pred, answer_instances)

            correct += (pred == answer_instances).sum().item()
            em5 += sum(answer_instances[i] in top5[i] for i in range(len(answer_instances)))
            em10 += sum(answer_instances[i] in top10[i] for i in range(len(answer_instances)))

            test_loss += criterion(logits.squeeze(-1), answer_instances).item()

            # Semantic accuracy
            pred_labels = [vocab[scene_id[i]][str(p.item())] for i, p in enumerate(pred)]
            semantic_correct += sum([pred_labels[i] == answer_labels[i] for i in range(len(pred_labels))])

    test_loss /= num_batches
    correct /= size
    accuracy = 100 * correct
    em5 = 100 * em5 / size
    em10 = 100 * em10 / size
    semantic_accuracy = 100 * semantic_correct / size

    print(f"Accuracy: {(accuracy):>0.1f}%, semantic_Accuracy: {(semantic_accuracy):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    # print(em5, em10)
    return accuracy, test_loss, logits, em5, em10, semantic_accuracy, attn_weights, word_tokens

def test_model(config_path, checkpoint=None):
    config = load_config(config_path)
    # wandb_logging(config=config, mode='init')
    
    train_dataset, train_loader = load_dataset(config, 'test', shuffle=False)
    # test_dataset, test_loader = load_dataset(config, 'test', shuffle=False)
    
    model = load_model(config)
    if checkpoint is not None:
        model.load_state_dict(torch.load(checkpoint))
        print(f'Model loaded from checkpoint: {checkpoint}')
    
    criterion = torch.nn.CrossEntropyLoss()

    vocab = load_json('vocab_all_scenes.json')
    accuracy, _, logits, em5, em10, semantic_accuracy, attn_weights, word_tokens = test_epoch(model, train_loader, criterion, vocab=vocab, verbose=True)
    # test_acc, test_loss, logits = test_epoch(model, test_loader, criterion, verbose=True)

    print('out')
    print(accuracy, em5, em10, semantic_accuracy)
    
    probs = F.softmax(logits, dim=1)
    # write_json(probs.squeeze().tolist(), "test_results0011.json", indent=1)


    q=10
    scene_id = train_dataset[0][4]
    gt_answer = train_dataset[0][2]
    path=f'TEST_VIZ_{scene_id}_{q}.png'
    pred = torch.argmax(logits.squeeze(-1), dim=1)
    visualize_attention(attn_weights.squeeze(0), word_tokens, scene_id, gt_answer, pred, path, probs)
    print('viz done')


if __name__ == "__main__":
    # train_model(config_path="config.yaml")
    
    # xxsXAttn_20k_sceneTrans_ckpt_20.pth
    # XAttn_20k_sceneTrans_ckpt_30.pth

    test_model(config_path="config.yaml", checkpoint="/cluster/scratch/akjaer/checkpoints/ATTENTION_PATTERN_EXP_ckpt_65.pth")
    # test_model(config_path="config.yaml")
    
    print('done')