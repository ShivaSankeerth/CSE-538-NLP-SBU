"""
author-gh: @adithya8
editor-gh: ykl7
"""

import math
from os import replace 

import numpy as np
import torch
import torch.nn as nn

sigmoid = lambda x: 1/(1 + torch.exp(-x))

class WordVec(nn.Module):
    def __init__(self, V, embedding_dim, loss_func, counts):
        super(WordVec, self).__init__()
        self.center_embeddings = nn.Embedding(num_embeddings=V, embedding_dim=embedding_dim)
        self.center_embeddings.weight.data.normal_(mean=0, std=1/math.sqrt(embedding_dim))
        self.center_embeddings.weight.data[self.center_embeddings.weight.data<-1] = -1
        self.center_embeddings.weight.data[self.center_embeddings.weight.data>1] = 1

        self.context_embeddings = nn.Embedding(num_embeddings=V, embedding_dim=embedding_dim)
        self.context_embeddings.weight.data.normal_(mean=0, std=1/math.sqrt(embedding_dim))
        self.context_embeddings.weight.data[self.context_embeddings.weight.data<-1] = -1 + 1e-10
        self.context_embeddings.weight.data[self.context_embeddings.weight.data>1] = 1 - 1e-10
        
        self.loss_func = loss_func
        self.counts = counts

    def forward(self, center_word, context_word):

        if self.loss_func == "nll":
            return self.negative_log_likelihood_loss(center_word, context_word)
        elif self.loss_func == "neg":
            return self.negative_sampling(center_word, context_word)
        else:
            raise Exception("No implementation found for %s"%(self.loss_func))
    
    def negative_log_likelihood_loss(self, center_word, context_word):
        ### TODO(students): start

        U = self.center_embeddings.weight[center_word]
        V = self.context_embeddings.weight[context_word]

        part_a = torch.sum(U*V,dim=1)
        part_b = torch.log(torch.sum(torch.exp(torch.matmul(V,U.T)),dim=1))

        loss = torch.mean(part_b-part_a)

        ### TODO(students): end

        return loss
    
    def negative_sampling(self, center_word, context_word):
        ### TODO(students): start
        U = self.center_embeddings.weight[center_word]
        V = self.context_embeddings.weight[context_word]

        # Number of negative samples to take
        k = 5
        
        # Randomly taking the negative samples using Unigram Distribution
        probs = self.counts/np.sum(self.counts)
        negative_samples = np.random.choice(len(self.counts),size=(len(center_word),k), replace=False,p=probs)
        neg_U = self.center_embeddings.weight[negative_samples]

        # Computing the loss by seperating the equation into part_a and part_b
        part_a = torch.log(sigmoid(torch.sum((U*V),dim=1)))

        U_shape = U.shape
        U = U.reshape((U_shape[0],U_shape[1],1))

        part_b = sigmoid(torch.sum(-torch.matmul(neg_U,U),dim=2))
        part_b = torch.sum(torch.log(part_b),dim=1)
        
        loss = torch.mean(-(part_a+part_b))

        ### TODO(students): end

        return loss

    def print_closest(self, validation_words, reverse_dictionary, top_k=8):
        print('Printing closest words')
        embeddings = torch.zeros(self.center_embeddings.weight.shape).copy_(self.center_embeddings.weight)
        embeddings = embeddings.data.cpu().numpy()

        validation_ids = validation_words
        norm = np.sqrt(np.sum(np.square(embeddings),axis=1,keepdims=True))
        normalized_embeddings = embeddings/norm
        validation_embeddings = normalized_embeddings[validation_ids]
        similarity = np.matmul(validation_embeddings, normalized_embeddings.T)
        for i in range(len(validation_ids)):
            word = reverse_dictionary[validation_words[i]]
            nearest = (-similarity[i, :]).argsort()[1:top_k+1]
            print(word, [reverse_dictionary[nearest[k]] for k in range(top_k)])            
