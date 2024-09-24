import torch
from transformers import CLIPModel, CLIPTokenizer

class CustomCLIPModel(CLIPModel):
    def get_word_features(
        self,
        input_ids= None,
        attention_mask=None,
        position_ids= None,
        output_attentions= None,
        output_hidden_states= None,
        return_dict = None,
    ) -> torch.FloatTensor:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        token_embeddings = text_outputs[0]  # Last hidden states (word-level embeddings), shape: (batch_size, max_seq_length, hidden_dim)
        word_features = token_embeddings @ self.text_projection.weight.T  # Apply the projection layer to each token embedding, shape: (batch_size, max_seq_length, output_dim)
        return word_features