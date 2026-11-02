# modules/rule_encoder.py

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class RuleEncoder(nn.Module):
    def __init__(self, 
                 model_name="bert-base-uncased",
                 embedding_dim=768,
                 use_symbolic_embedding=True):
        super().__init__()
        self.use_symbolic_embedding = use_symbolic_embedding

        # Textual encoder: BERT or similar
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.text_encoder = AutoModel.from_pretrained(model_name)
        self.embedding_dim = embedding_dim

        if use_symbolic_embedding:
            # Optionally learn symbolic-type embeddings: [IF], [THEN], [AND], etc.
            self.rule_tokens = ["IF", "THEN", "AND", ">", "<", "==", ">=", "<=", "AND_THEN", "MULTI_ACTION"]
            self.token2idx = {tok: idx for idx, tok in enumerate(self.rule_tokens)}
            self.symbol_embedding = nn.Embedding(len(self.rule_tokens), embedding_dim)

    def forward(self, rule: str, return_type='tensor'):
        """
        Encode a rule (string) to vector representation.
        rule: a structured rule string (e.g. "IF speed > 120 THEN risk ↑")
        return_type: 'tensor' or 'dict' for debug
        """
        # Step 1: BERT embedding of rule string
        inputs = self.tokenizer(rule, return_tensors="pt", padding=True, truncation=True)
        outputs = self.text_encoder(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token embedding

        # Step 2: Symbolic structure embedding (optional)
        if self.use_symbolic_embedding:
            symbol_indices = self._extract_symbol_indices(rule)  # list[int]
            symbol_tensor = torch.tensor(symbol_indices).unsqueeze(0)  # [1, num_symbols]
            symbol_embed = self.symbol_embedding(symbol_tensor).mean(dim=1)  # average symbolic embedding
            final_embedding = cls_embedding + symbol_embed
        else:
            final_embedding = cls_embedding

        if return_type == 'dict':
            return {
                'text_embedding': cls_embedding.detach(),
                'symbolic_embedding': symbol_embed.detach() if self.use_symbolic_embedding else None,
                'final_embedding': final_embedding.detach()
            }

        return final_embedding  # tensor of shape [1, embedding_dim]

    def _extract_symbol_indices(self, rule: str):
        """
        Extract symbolic tokens used in the rule.
        E.g., "IF speed > 120 AND visibility < 0.3 THEN risk ↑" → ["IF", ">", "AND", "<", "THEN"]
        """
        tokens = []
        for symbol in self.rule_tokens:
            if symbol in rule:
                tokens.append(self.token2idx[symbol])
        return tokens if tokens else [self.token2idx["IF"]]  # default to [IF] if none found
