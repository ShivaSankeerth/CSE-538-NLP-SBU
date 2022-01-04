# inbuilt lib imports:
from typing import Dict
import math

# external libs
import torch
import torch.nn as nn
import numpy as np
from scipy.stats import truncnorm


# project imports

def truncated_normal(size, stddev):
    threshold = 2 * stddev
    values = truncnorm.rvs(-threshold, threshold, size=size)
    return values


class CubicActivation(nn.Module):
    """
    Cubic activation as described in the paper.
    """

    def __init__(self) -> None:
        super(CubicActivation, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

    def forward(self, vector: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        vector : ``torch.Tensor``
            hidden vector of dimension (batch_size, hidden_dim)

        Returns tensor after applying the activation
        """
        # TODO(Students) Start  
        
        return torch.pow(vector, torch.full(vector.size(), 3.0).to(self.device))
       

        # TODO(Students) End


class DependencyParser(nn.Module):
    def __init__(self,
                 embedding_dim: int,
                 vocab_size: int,
                 num_tokens: int,
                 hidden_dim: int,
                 num_transitions: int,
                 regularization_lambda: float,
                 trainable_embeddings: bool,
                 device: str,
                 activation_name: str = "cubic") -> None:
        """
        This model defines a transition-based dependency parser which makes
        use of a classifier powered by a neural network. The neural network
        accepts distributed representation inputs: dense, continuous
        representations of words, their part of speech tags, and the labels
        which connect words in a partial dependency parse.

        This is an implementation of the method described in

        Danqi Chen and Christopher Manning.
        A Fast and Accurate Dependency Parser Using Neural Networks. In EMNLP 2014.

        Parameters
        ----------
        embedding_dim : ``str``
            Dimension of word embeddings
        vocab_size : ``int``
            Number of words in the vocabulary.
        num_tokens : ``int``
            Number of tokens (words/pos) to be used for features
            for this configuration.
        hidden_dim : ``int``
            Hidden dimension of feedforward network
        num_transitions : ``int``
            Number of transitions to choose from.
        regularization_lambda : ``float``
            Regularization loss fraction lambda as given in paper.
        trainable_embeddings : `bool`
            Is the embedding matrix trainable or not.
        """
        super(DependencyParser, self).__init__()
        self._regularization_lambda = regularization_lambda

        if activation_name == "cubic":
            self._activation = CubicActivation()
        elif activation_name == "sigmoid":
            self._activation = torch.sigmoid
        elif activation_name == "tanh":
            self._activation = torch.tanh
        else:
            raise Exception(f"activation_name: {activation_name} is from the known list.")

        # Trainable Variables
        # TODO(Students) Start

        self.tensor_device = device
        self.embeddings = torch.nn.Embedding(vocab_size, embedding_dim).to(self.tensor_device)
        
        torch.nn.init.uniform_(self.embeddings.weight, -0.01, 0.01)

        self.hidden_layer_weights = truncated_normal([num_tokens * embedding_dim, hidden_dim],
                             1 / math.sqrt(num_tokens * embedding_dim))
        self.hidden_layer_weights = torch.from_numpy(self.hidden_layer_weights.astype(np.float32)).to(self.tensor_device)
            
        
        self.weight_sm = truncated_normal([hidden_dim, num_transitions],
                             1 / math.sqrt(hidden_dim))
        self.weight_sm = torch.from_numpy(self.weight_sm.astype(np.float32)).to(self.tensor_device)


        self.hidden_layer_bias = truncated_normal([hidden_dim],1 / math.sqrt(hidden_dim))
        self.hidden_layer_bias = torch.from_numpy(self.hidden_layer_bias.astype(np.float32)).to(self.tensor_device)



        self.hidden_layer_bias = torch.nn.Parameter(self.hidden_layer_bias)
        self.hidden_layer_weights = torch.nn.Parameter(self.hidden_layer_weights)
        self.weight_sm = torch.nn.Parameter(self.weight_sm)


        # TODO(Students) End

    def forward(self,
                inputs: torch.Tensor,
                labels: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass of Dependency Parser.

        Parameters
        ----------
        inputs : ``torch.Tensor``
            Tensorized version of the batched input text. It is of shape:
            (batch_size, num_tokens) and entries are indices of tokens
            in to the vocabulary. These tokens can be word or pos tag.
            Each row corresponds to input features a configuration.
        labels : ``torch.Tensor``
            Tensor of shape (batch_size, num_transitions)
            Each row corresponds to the correct transition that
            should be made in the given configuration.

        Returns
        -------
        An output dictionary consisting of:
        logits : ``torch.Tensor``
            A tensor of shape ``(batch_size, num_transitions)`` representing
            logits (unnormalized scores) for the labels for every instance in batch.
        loss : ``torch.float32``
            If input has ``labels``, then mean loss for the batch should
            be computed and set to ``loss`` key in the output dictionary.

        """
        # TODO(Students) Start

        
        reshaped_embeddings = torch.reshape(self.embeddings(inputs), [inputs.shape[0], -1]).to(self.tensor_device)
        # print(type(reshaped_embeddings),type(self.hidden_layer_weights))

        hidden_layer_op = torch.add(torch.matmul(reshaped_embeddings, self.hidden_layer_weights), self.hidden_layer_bias).to(self.tensor_device)
        hidden_layer_op= self._activation(hidden_layer_op)
        logits = torch.matmul(hidden_layer_op, self.weight_sm)

        # TODO(Students) End
        output_dict = {"logits": logits}

        if labels is not None:
            output_dict["loss"] = self.compute_loss(logits, labels)
        
        return output_dict

    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.float32:
        """
        Parameters
        ----------
        logits : ``torch.Tensor``
            A tensor of shape ``(batch_size, num_transitions)`` representing
            logits (unnormalized scores) for the labels for every instance in batch.

        Returns
        -------
        loss : ``torch.float32``
            If input has ``labels``, then mean loss for the batch should
            be computed and set to ``loss`` key in the output dictionary.

        """
        # TODO(Students) Start

        valid_transitions = torch.where(labels == -1, torch.zeros(labels.shape, dtype=torch.float32).to(self.tensor_device),
                                           torch.ones(labels.shape, dtype=torch.float32).to(self.tensor_device))

        labels = labels.type(torch.FloatTensor).to(self.tensor_device)
        labels_toform = torch.multiply(labels, valid_transitions).to(self.tensor_device)

        logits_sm = torch.divide(torch.exp(logits * valid_transitions), torch.sum(torch.exp(logits * valid_transitions), axis=1, keepdims=True))

        loss1 = -torch.mean(torch.sum(labels_toform * torch.log(logits_sm + 10e-10), axis=1, keepdims=True))

        # loss_func = nn.CrossEntropyLoss()
        # new_bias = self.hidden_layer_bias.reshape(self.hidden_layer_bias.shape[0], 1)
        # loss2 = (self._regularization_lambda / 2.0) * (
        #         loss_func(self.hidden_layer_weights, self.hidden_layer_weights) + loss_func(self.weight_sm,
        #                                                                    self.weight_sm) + loss_func(new_bias,
        #                                                                                          self.hidden_layer_bias))

        # regularization_loss = loss2 * self._regularization_lambda * (1.0/2.0)

        l2 = self.hidden_layer_weights.norm(2) + self.weight_sm.norm(2) + self.hidden_layer_bias.norm(2) + self.embeddings.weight.norm(2)
        regularization = l2* self._regularization_lambda * (1.0/2.0)

        loss = loss1 + regularization
       

        # TODO(Students) End
        return loss
