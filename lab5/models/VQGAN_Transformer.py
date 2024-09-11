import torch 
import torch.nn as nn
import yaml
import os
import math
import numpy as np
from .VQGAN import VQGAN
from .Transformer import BidirectionalTransformer


#TODO2 step1: design the MaskGIT model
class MaskGit(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.vqgan = self.load_vqgan(configs['VQ_Configs'])
    
        self.num_image_tokens = configs['num_image_tokens']
        self.mask_token_id = configs['num_codebook_vectors']
        self.choice_temperature = configs['choice_temperature']
        self.gamma = self.gamma_func(configs['gamma_type'])
        self.transformer = BidirectionalTransformer(configs['Transformer_param'])

    def load_transformer_checkpoint(self, load_ckpt_path):
        self.transformer.load_state_dict(torch.load(load_ckpt_path, weights_only=True))

    @staticmethod
    def load_vqgan(configs):
        cfg = yaml.safe_load(open(configs['VQ_config_path'], 'r'))
        model = VQGAN(cfg['model_param'])
        model.load_state_dict(torch.load(configs['VQ_CKPT_path'], weights_only=True), strict=True) 
        model = model.eval()
        return model
    
##TODO2 step1-1: input x fed to vqgan encoder to get the latent and zq
    @torch.no_grad()
    def encode_to_z(self, x):
        zq, z_indices, _ = self.vqgan.encode(x)
        return zq, z_indices.reshape(zq.shape[0], -1)
    
##TODO2 step1-2:    
    def gamma_func(self, mode="cosine"):
        """Generates a mask rate by scheduling mask functions R.

        Given a ratio in [0, 1), we generate a masking ratio from (0, 1]. 
        During training, the input ratio is uniformly sampled; 
        during inference, the input ratio is based on the step number divided by the total iteration number: t/T.
        Based on experiements, we find that masking more in training helps.
        
        ratio:   The uniformly sampled ratio [0, 1) as input.
        Returns: The mask rate (float).

        """
        if mode == "linear":
            return lambda r: 1 - r
        elif mode == "cosine":
            return lambda r: 0.5 * (1 + np.cos(np.pi * r))
        elif mode == "square":
            return lambda r: 1 - r**2
        else:
            raise NotImplementedError("Unknown mode for gamma_func")

##TODO2 step1-3:            
    def forward(self, x):
        # x: (b, c, h, w)
        _, z_indices = self.encode_to_z(x)

        mask_token = torch.ones(z_indices.shape, device = z_indices.device).long() * self.mask_token_id 
        
        ratio = np.random.uniform(0, 1)
        mask = torch.bernoulli(ratio * torch.ones(z_indices.shape, device = z_indices.device)).bool()
        
        input_indices = torch.where(mask, mask_token, z_indices)
        logits = self.transformer(input_indices)

        return logits, z_indices
    
##TODO3 step1-1: define one iteration decoding   
    @torch.no_grad()
    def inpainting(self, z_indices, mask_b, mask_num, ratio, mask_func):

        z_masked_indices = torch.where(mask_b, torch.tensor(self.mask_token_id).to(mask_b.device), z_indices)
        logits = self.transformer(z_masked_indices)
        #Apply softmax to convert logits into a probability distribution across the last dimension.
        logits = torch.softmax(logits, dim=-1)

        #FIND MAX probability for each token value
        #At the end of the decoding process, add back the original token values that were not masked to the predicted tokens
        z_indices_predict_prob, z_indices_predict = torch.max(logits, dim=-1)
        z_indices_predict = torch.where(mask_b, z_indices_predict, z_indices)

        mask_ratio = self.gamma_func(mask_func)(ratio)

        #predicted probabilities add temperature annealing gumbel noise as confidence
        g = torch.distributions.gumbel.Gumbel(0, 1).sample(z_indices_predict_prob.shape).to(z_indices_predict_prob.device)
        temperature = self.choice_temperature * (1 - mask_ratio)
        confidence = z_indices_predict_prob + temperature * g
        
        #hint: If mask is False, the probability should be set to infinity, so that the tokens are not affected by the transformer's prediction
        #sort the confidence for the rank 
        confidence = torch.where(mask_b, confidence, torch.tensor(float('inf')).to(mask_b.device))

        mask_ratio = 0 if mask_ratio < 1e-8 else mask_ratio
        mask_len = torch.floor(mask_num * mask_ratio).long()
        _, confidence_sorted_indice = torch.topk(confidence, mask_len, largest = False)
        
        #define how much the iteration remain predicted tokens by mask scheduling
        mask_bc = torch.zeros_like(mask_b)
        mask_bc.index_fill_(1, confidence_sorted_indice.squeeze(), 1)
        torch.bitwise_and(mask_bc, mask_b, out = mask_bc)
        return z_indices_predict, mask_bc
        
    
__MODEL_TYPE__ = {
    "MaskGit": MaskGit
}
