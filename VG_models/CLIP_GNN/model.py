import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from transformers import CLIPTokenizer, CLIPTextModel, CLIPModel
from customCLIP_model import CustomCLIPModel
from torch_geometric.utils import to_dense_batch
import matplotlib.pyplot as plt
import seaborn as sns
import re
from utils import load_json, clip_embed

device = "cuda" if torch.cuda.is_available() else "cpu"

class GNN(nn.Module):
    def __init__(self, gnn_type):
        super().__init__()
        self.conv1 = GCNConv(512, 512) # CLIP embeddings are 512-dimensional

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        # x = F.leaky_relu(x) # Leaky relu used to avoid 0-division when computing cosine similarity
        
        # x = F.dropout(x, training=self.training)
        # x = self.conv2(x, edge_index)
        # x = F.leaky_relu(x)
        
        return x

class SceneEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, num_encoder_layers):
        super(SceneEncoder, self).__init__()
        self.scene_transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=ff_dim,
                batch_first=True
            ),
            num_layers=num_encoder_layers
        )
    
    def forward(self, nodes, src_key_padding_mask):
        node_embeddings = self.scene_transformer_encoder(
            src=nodes,
            src_key_padding_mask=src_key_padding_mask
        )
        return node_embeddings

class QuestionEncoder(nn.Module):
    """
    Word-level encoder: 
    """
    def __init__(self, embed_dim, num_heads, num_encoder_layers, ff_dim, max_seq_length):
        super(QuestionEncoder, self).__init__()
        """------- CLIP EMBEDDING -------"""
        self.max_seq_length = max_seq_length
        self.tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32')
        # self.clip_model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
        self.clip_model = CustomCLIPModel.from_pretrained('openai/clip-vit-base-patch32')
        
        """------- TRANSFORMER ENCODER -------"""
        self.question_transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=ff_dim,
                batch_first=True
            ),
            num_layers=num_encoder_layers
        )
        
    def forward(self, question):
        """------- WORD-LEVEL CLIP EMBEDDINGS -------"""
        tokens = self.tokenizer(
            question,
            padding='max_length',
            max_length=self.max_seq_length,
            return_tensors='pt',
            truncation=True
        )
        # tokens['input_ids'].shape = torch.Size([batch_size, max_seq_length])
        # tokens['attention_mask'].shape = torch.Size([batch_size, max_seq_length])
        
        decoded_tokens = self.tokenizer.decode(tokens['input_ids'][0], skip_special_tokens=False)
        word_tokens = re.split(r'(?<= )|(?=\?)|(?<=[>])', decoded_tokens)[:-1]
        
        with torch.no_grad():
            word_embeddings = self.clip_model.get_word_features(**tokens) # torch.Size([batch_size, max_seq_length, embed_dim=512])
            # word_embeddings = self.clip_model.get_text_features(**tokens).last_hidden_state
        # normalize the embeddings
        word_embeddings = F.normalize(word_embeddings, p=2, dim=-1)

        """------- TRANSFORMER ENCODER FOR WORDS IN QUESTION -------"""
        key_padding_mask = (1 - tokens['attention_mask'].float()).to(device) # torch.Size([batch_size, max_seq_length])
        src_mask = torch.triu(torch.ones(self.max_seq_length, self.max_seq_length) * float('-inf'), diagonal=1) # torch.Size([max_seq_length, max_seq_length])
        # word_embeddings = self.question_transformer_encoder(
        #     src=word_embeddings,
        #     src_key_padding_mask=key_padding_mask,
        #     mask=src_mask
        # )
        
        return word_embeddings, key_padding_mask, word_tokens

class CrossAttentionBlock(nn.Module):
    def __init__(self, num_heads, ff_dim, dropout, embed_dim):
        super(CrossAttentionBlock, self).__init__()
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.layer_norm1 = nn.LayerNorm(normalized_shape=embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim)
        )
        self.layer_norm2 = nn.LayerNorm(normalized_shape=embed_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, nodes, questions, key_padding_mask):
        # questions = [batch_size, max_seq_len, embed_dim]
        # nodes = [batch_size, num_nodes, embed_dim]
        # key_padding_mask = [batch_size, max_seq_len]

        # Cross-Attention
        attn_output, attn_weights = self.multihead_attention(
            query=nodes,
            key=questions,
            value=questions,
            key_padding_mask=key_padding_mask,
            need_weights=True
        )
        
        nodes = self.layer_norm1(nodes + self.dropout1(attn_output))
        ff_output = self.feed_forward(nodes)
        nodes = self.layer_norm2(nodes + self.dropout2(ff_output))
        
        return nodes, attn_weights


# class SequentialCrossAttention(nn.Sequential):
#     def forward(self, *inputs):
#         nodes, words, mask = inputs
#         for module in self._modules.values():
#             nodes = module(nodes, words, mask)
#         return nodes

class SequentialCrossAttention(nn.Sequential):
    def forward(self, *inputs):
        nodes, words, mask = inputs
        attn_weights_list = []
        
        for module in self._modules.values():
            nodes, attn_weights = module(nodes, words, mask)
            attn_weights_list.append(attn_weights)
        
        # Stack attn_weights across layers
        stacked_attn_weights = torch.stack(attn_weights_list)  # Shape: [num_layers, batch_size, num_nodes, max_seq_len]
        avg_attn_weights = torch.mean(stacked_attn_weights, dim=0)  # Shape: [batch_size, num_nodes, max_seq_len]
        return nodes, avg_attn_weights

class VQAModel_attn(nn.Module):
    def __init__(
        self,
        # GCN Scene graph encoding
        gnn_type,            
        # Question encoding
        embed_dim,
        num_heads,
        num_encoder_layers,
        ff_dim,
        max_seq_length,
        # Cross attention
        xattn_num_layers,
        xattn_ff_dim,
        xattn_num_heads,
        dropout,
        ff_dim_downsize
    ):
        super(VQAModel_attn, self).__init__()

        self.gnn = GNN(gnn_type)
        
        self.scene_encoder = SceneEncoder(
            embed_dim=512,
            num_heads=4,
            ff_dim=512,
            num_encoder_layers=2
        )

        self.question_encoder = QuestionEncoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_encoder_layers=num_encoder_layers,
            ff_dim=ff_dim,
            max_seq_length=max_seq_length
        )

        # self.downsizing_mlp = nn.Sequential(
        #     nn.Linear(512, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 32)
        # )

        self.cross_attention_decoder = SequentialCrossAttention(
            *[CrossAttentionBlock(num_heads=xattn_num_heads, ff_dim=xattn_ff_dim, dropout=dropout, embed_dim=embed_dim) for _ in range(xattn_num_layers)]
            )
        
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, ff_dim_downsize),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim_downsize, 1)
        )
        

    def forward(self, graph_data, question_raw):
        """------- NODE ENCODING -------"""
        # node_features = self.gnn(graph_data) # torch.Size([all_nodes_in_batch, embed_dim=512])
        node_features = graph_data.x

        # Add batch dimension to GNN output (and pad small graphs)
        node_features, node_padding_mask = to_dense_batch(
            x=node_features,
            batch=graph_data.batch,
            max_num_nodes=max([data.num_nodes for data in graph_data.to_data_list()]),
            batch_size=graph_data.num_graphs
        ) # torch.Size([batch_size, max_num_nodes, embed_dim=512])

        # node_features = self.scene_encoder(node_features, ~node_padding_mask)

        """------- QUESTION ENCODING -------"""
        word_embeddings, seq_padding_mask, word_tokens = self.question_encoder(question_raw) # torch.Size([batch_size, max_seq_len, embed_dim=512])
        
        """------- DOWNSIZE EMBEDDINGS -------"""
        # node_features = self.downsizing_mlp(node_features)
        # word_embeddings = self.downsizing_mlp(word_embeddings)

        """------- NODES-TO-QUESTION CROSS-ATTENTION -------"""
        node_features, avg_attn_weights = self.cross_attention_decoder(node_features, word_embeddings, seq_padding_mask)
        
        """------- MLP / LINEAR LAYER -------"""
        node_features = self.mlp(node_features)
        # Mask out padded nodes
        node_padding_mask = node_padding_mask.unsqueeze(-1).expand(-1, -1, 1) # torch.Size([batch_size, max_num_nodes, 1])
        node_features[~node_padding_mask] = float('-inf')
        logits = node_features

        return logits, avg_attn_weights, word_tokens

"""

OTHER MODELS

"""
class CLIP_GNN(nn.Module):
    def __init__(
        self,
        max_seq_length,
        gnn_type,
    ):
        super(CLIP_GNN, self).__init__()
        self.gnn = GNN(gnn_type)
        self.max_seq_length = max_seq_length
        self.tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32')
        self.clip_model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')

    def forward(self, graph_data, question):
        """------- NODE ENCODING -------"""
        node_features = self.gnn(graph_data) # torch.Size([all_nodes_in_batch, embed_dim=512])
        node_features =  node_features / node_features.norm(dim=1, keepdim=True)

        # Add batch dimension to GNN output (and pad small graphs)
        node_features, node_padding_mask = to_dense_batch(
            x=node_features,
            batch=graph_data.batch,
            max_num_nodes=max([data.num_nodes for data in graph_data.to_data_list()]),
            batch_size=graph_data.num_graphs
        ) # torch.Size([batch_size, max_num_nodes, embed_dim=512])

        """------- QUESTION ENCODING -------"""
        tokens = self.tokenizer(
            question,
            padding='max_length',
            max_length=self.max_seq_length,
            return_tensors='pt',
            truncation=True
        )
        # tokens['input_ids'].shape = torch.Size([batch_size, max_seq_length])
        # tokens['attention_mask'].shape = torch.Size([batch_size, max_seq_length])

        with torch.no_grad():
            question_features = self.clip_model.get_text_features(**tokens) # torch.Size([batch_size, embed_dim])
        question_features = question_features / question_features.norm(dim=1, keepdim=True) # torch.Size([batch_size, embed_dim])

        """------- COSINE SIMILARITY -------"""      
        logits = torch.matmul(node_features, question_features.unsqueeze(1).transpose(1, 2)) * node_padding_mask.unsqueeze(-1) # torch.Size([batch_size, max_num_nodes,1])
        return logits, None, None
    
class Baseline(nn.Module):
    def __init__(
        self,
        max_seq_length
    ):
        super(Baseline, self).__init__()
        self.max_seq_length = max_seq_length
        self.tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32')
        self.clip_model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')

    def forward(self, graph_data, question):
        node_features = graph_data.x # torch.Size([all_nodes_in_batch, embed_dim=512])
        node_features =  node_features / node_features.norm(dim=1, keepdim=True)

        # Add batch dimension to GNN output (and pad small graphs)
        node_features, node_padding_mask = to_dense_batch(
            x=node_features,
            batch=graph_data.batch,
            max_num_nodes=max([data.num_nodes for data in graph_data.to_data_list()]),
            batch_size=graph_data.num_graphs
        ) # torch.Size([batch_size, max_num_nodes, embed_dim=512])

        """------- QUESTION ENCODING -------"""
        tokens = self.tokenizer(
            question,
            padding='max_length',
            max_length=self.max_seq_length,
            return_tensors='pt',
            truncation=True
        )

        with torch.no_grad():
            question_features = self.clip_model.get_text_features(**tokens) # torch.Size([batch_size, embed_dim])
        question_features = question_features / question_features.norm(dim=1, keepdim=True) # torch.Size([batch_size, embed_dim])

        """------- COSINE SIMILARITY -------"""      
        logits = torch.matmul(node_features, question_features.unsqueeze(1).transpose(1, 2)) * node_padding_mask.unsqueeze(-1) # torch.Size([batch_size, max_num_nodes,1])
        # logits = torch.matmul(node_features, question_features.unsqueeze(1).transpose(1, 2)) # torch.Size([batch_size, max_num_nodes,1])
        return logits, None, None


"""
ATTENTION VISUALIZATION

"""
def visualize_attention(attn_weights, question_tokens, scene_id, gt_answer, pred, path, probs):
    
    def get_dynamic_fontsize(y_labels, fig_height):
        max_chars = max(len(label) for label in y_labels) # max number of characters in y_labels
        available_height_per_label = fig_height / len(y_labels) # avail height for each label
        fontsize = min(available_height_per_label * 1.5, 14)  # heuristic measure, 10 is the max fontsize limit
        return max(fontsize, 11)  # ensure minimum fontsize 5

    # create labels for y-axis
    y_labels_map=load_json('vocab_all_scenes.json')[scene_id]
    probs=probs.squeeze(0).squeeze(1).detach().numpy()
    y_labels=[f'{k}: {y_labels_map[str(k)]} ({p:.2f})' for k, p in zip(y_labels_map.keys(), probs)]

    attn_weights = attn_weights.detach().numpy()  # Shape: [num_nodes, max_seq_len]

    f = question_tokens.index('<|endoftext|>', 1) + 1
    question_tokens = question_tokens[:f]
    attn_weights = attn_weights[:, :f]

    fig_height = 10
    fontsize = get_dynamic_fontsize(y_labels, fig_height)

    plt.figure(figsize=(8, 10))
    vmin=0
    vmax=0.4
    ax = sns.heatmap(attn_weights, xticklabels=question_tokens, yticklabels=y_labels, cmap='viridis', vmin=vmin, vmax=vmax)
    
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=fontsize)
    x_fontsize = fontsize
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=x_fontsize)
    # highlight correct answer
    for label in ax.get_yticklabels():
        if label.get_text() == y_labels[gt_answer]:
            label.set_color('green')
            label.set_fontweight('bold')
    
    # highlight incorrect prediction (if prediction was incorrect)
    if pred != gt_answer:
        for label in ax.get_yticklabels():
            if label.get_text() == y_labels[pred]:
                label.set_color('red')
                label.set_fontweight('bold')
    
    plt.savefig(path, format='png', dpi=300, bbox_inches='tight')
    plt.close()