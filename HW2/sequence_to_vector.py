'''
author: Sounak Mondal
'''

# std lib imports
from typing import Dict

# external libs
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

torch.manual_seed(1337)

def flatten(t):
    return [item for item in t]

class SequenceToVector(nn.Module):
    """
    It is an abstract class defining SequenceToVector enocoder
    abstraction. To build you own SequenceToVector encoder, subclass
    this.

    Parameters
    ----------
    input_dim : ``str``
        Last dimension of the input input vector sequence that
        this SentenceToVector encoder will encounter.
    """
    def __init__(self,
                 input_dim: int) -> 'SequenceToVector':
        super(SequenceToVector, self).__init__()
        self._input_dim = input_dim

    def forward(self,
             vector_sequence: torch.Tensor,
             sequence_mask: torch.Tensor,
             training=False) -> Dict[str, torch.Tensor]:
        """
        Forward pass of Main Classifier.

        Parameters
        ----------
        vector_sequence : ``torch.Tensor``
            Sequence of embedded vectors of shape (batch_size, max_tokens_num, embedding_dim)
        sequence_mask : ``torch.Tensor``
            Boolean tensor of shape (batch_size, max_tokens_num). Entries with 1 indicate that
            token is a real token, and 0 indicate that it's a padding token.
        training : ``bool``
            Whether this call is in training mode or prediction mode.
            This flag is useful while applying dropout because dropout should
            only be applied during training.

        Returns
        -------
        An output dictionary consisting of:
        combined_vector : torch.Tensor
            A tensor of shape ``(batch_size, embedding_dim)`` representing vector
            compressed from sequence of vectors.
        layer_representations : torch.Tensor
            A tensor of shape ``(batch_size, num_layers, embedding_dim)``.
            For each layer, you typically have (batch_size, embedding_dim) combined
            vectors. This is a stack of them.
        """
        # ...
        # return {"combined_vector": combined_vector,
        #         "layer_representations": layer_representations}
        raise NotImplementedError


class DanSequenceToVector(SequenceToVector):
    """
    It is a class defining Deep Averaging Network based Sequence to Vector
    encoder. You have to implement this.

    Parameters
    ----------
    input_dim : ``str``
        Last dimension of the input input vector sequence that
        this SentenceToVector encoder will encounter.
    num_layers : ``int``
        Number of layers in this DAN encoder.
    dropout : `float`
        Token dropout probability as described in the paper.
    """
    def __init__(self, input_dim: int, num_layers: int, dropout: float = 0.2, device = 'cpu'):
        super(DanSequenceToVector, self).__init__(input_dim)
        # TODO(students): start
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.device = device
        
        self.layers = torch.nn.ModuleList()
        
        for i in range(self.num_layers-1):
            self.layers.append(nn.Linear(in_features=self.input_dim,out_features=self.input_dim))
            self.layers.append(nn.ReLU())
        
        self.layers.append(nn.Linear(self.input_dim,self.input_dim))

        # TODO(students): end

    def forward(self,
             vector_sequence: torch.Tensor,
             sequence_mask: torch.Tensor,
             training=False) -> torch.Tensor:
        # TODO(students): start
        shape = vector_sequence.shape
        
        sequence_mask = torch.reshape(sequence_mask, (shape[0], shape[1], 1))
        vector_sequence *= sequence_mask

        num_words = torch.sum(sequence_mask, dim=1)
        
        if training:
            # randomly dropping word tokens’ entire word embeddings by creating a vector using bernoulli distribution
            # using n = 1 and p = 0.5 changes the binomial distribution into a bernouli distribution
            # source: https://stackoverflow.com/a/47014668
            bernoulli_vector = np.random.binomial(n=1,p=0.5,size=(shape[0], shape[1], 1))
            word_drop_mask = torch.from_numpy(bernoulli_vector)
            word_drop_mask = word_drop_mask >= self.dropout
            word_drop_mask = word_drop_mask.type(torch.FloatTensor)
            num_words = torch.sum(word_drop_mask * sequence_mask, dim=1)

            vector_sequence *= word_drop_mask
        
        combined_vector = torch.div(torch.sum(vector_sequence, dim=1), num_words)
        layer_reps = []
        for layer in self.layers:  
            combined_vector = layer(combined_vector)
            layer_reps.append(combined_vector)
        
        # Stacking sentence representations of every layer into a tensor
        layer_reps = torch.stack(layer_reps, dim=1)

        # TODO(students): end
        return {"combined_vector": combined_vector,
                "layer_representations": layer_reps}


class GruSequenceToVector(SequenceToVector):
    """
    It is a class defining GRU based Sequence To Vector encoder.
    You have to implement this.

    Parameters
    ----------
    input_dim : ``str``
        Last dimension of the input input vector sequence that
        this SentenceToVector encoder will encounter.
    num_layers : ``int``
        Number of layers in this GRU-based encoder. Note that each layer
        is a GRU encoder unlike DAN where they were feedforward based.
    """
    def __init__(self, input_dim: int, num_layers: int, device = 'cpu'):
        super(GruSequenceToVector, self).__init__(input_dim)
        # TODO(students): start

        self.input_dim = input_dim
        self.num_layers = num_layers

        self.model = nn.GRU(input_size=self.input_dim,num_layers=self.num_layers,hidden_size=self.input_dim)
        
        # TODO(students): end

    def forward(self,
             vector_sequence: torch.Tensor,
             sequence_mask: torch.Tensor,
             training=False) -> torch.Tensor:
        
        # TODO(students): start
        lengths = sequence_mask.sum(dim=1)
        packed_seq = torch.nn.utils.rnn.pack_padded_sequence(vector_sequence,lengths=lengths,batch_first=True,enforce_sorted=False)
        output, final_hidden_states = self.model(packed_seq)
        layer_representations = final_hidden_states.transpose(0,1)
        combined_vector = final_hidden_states[-1]
        # TODO(students): end
        return {"combined_vector": combined_vector,
                "layer_representations": layer_representations}
