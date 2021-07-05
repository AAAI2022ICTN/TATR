import torch
import torch.nn as nn
import torch.nn.functional as F
from myUtils import cp
import logging
import numpy as np

def myLoss(y_true, y_pred):
    loss = multi_label_cross_entropy(y_true, y_pred)
    return loss
    
def multi_label_cross_entropy(y_true, y_pred):
    temp = -1*y_pred*y_true
    loss = torch.mean(torch.sum(temp, 1))
    return loss       


class myModel(torch.nn.Module):

    def __init__(self, opt):
        super(myModel, self).__init__()
        
        self.vocab_size = opt.vocab_size
        self.drop_rate = opt.drop_rate
        self.num_layers = opt.num_layers
        self.n_classes = opt.label_num
        self.batch_size = opt.batch_size
        self.hidden_size = opt.hidden_size
        self.embed_size = opt.emb_size
        self.bidirectional = opt.bidirectional
        self.num_directions = 2 if opt.bidirectional else 1
        self.hidden_size = self.hidden_size* self.num_directions
        self.att_dim = opt.att_dim
        
        self.topic_num = opt.topic_num
        self.attn_mode = opt.attn_mode   
        self.use_topic_represent = opt.use_topic_represent   

        self.embeddings = torch.nn.Embedding(self.vocab_size,self.embed_size)
        self.rnn = torch.nn.GRU(input_size=self.embed_size, hidden_size=opt.hidden_size, num_layers=self.num_layers, batch_first=True, bidirectional=self.bidirectional, dropout=self.drop_rate)         
        self.W1 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.W11 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.W2 = torch.nn.Linear(self.hidden_size, 1)
        self.W22 = torch.nn.Linear(self.hidden_size, 1)
        self.W23 = torch.nn.Linear(self.hidden_size, 1)
        self.W24 = torch.nn.Linear(self.hidden_size, 1)
        self.W3 = torch.nn.Linear(self.topic_num, self.hidden_size)
        self.W4 = torch.nn.Linear(self.hidden_size +self.topic_num, self.hidden_size)
        self.W5 = torch.nn.Linear(self.hidden_size, self.att_dim)
        self.W51 = torch.nn.Linear(self.hidden_size, self.att_dim)
        self.W6 = torch.nn.Linear(self.hidden_size, self.n_classes, bias=False)
        self.W7 = torch.nn.Linear(256, 256)
        self.W8 = torch.nn.Linear(256, 1)       
        nn.init.xavier_uniform_(self.W7.weight)
        nn.init.xavier_uniform_(self.W8.weight) 
        self.W9 = torch.nn.Linear(self.topic_num, opt.max_src_len)
               
        self.output_layer = torch.nn.Linear(self.hidden_size, self.n_classes)
        self.output_layer2 = torch.nn.Linear(self.hidden_size*2, self.n_classes)
        self.output_layer3 = torch.nn.Linear(self.topic_num, self.n_classes)
        
        self.layer_norm = torch.nn.LayerNorm(self.hidden_size)
        self.batch_norm = torch.nn.BatchNorm1d(self.embed_size)    
        
        #MLA
        self.attention = nn.Linear(self.hidden_size, self.n_classes, bias=False)
        nn.init.xavier_uniform_(self.attention.weight)
        opt.linear_size = [256]#[512, 256]
        self.linear_size = [self.hidden_size] + opt.linear_size 
        self.output_size = 1
        self.linear = nn.ModuleList(nn.Linear(in_s, out_s) for in_s, out_s in zip(self.linear_size[:-1], self.linear_size[1:]))    
        for linear in self.linear:
            nn.init.xavier_uniform_(linear.weight)
        self.output = nn.Linear(self.linear_size[-1], self.output_size)
        nn.init.xavier_uniform_(self.output.weight)  
        
        
    def forward(self, x, topic_represent):#topic_represent [batch_size, topic_num]
        embeddings = self.embeddings(x)  #[batch, src_len, embed_size]

        #embeddings = self.batch_norm(embeddings)
        memory_bank, encoder_final_state = self.rnn(embeddings)
        # ([batch, seq_len, num_directions*hidden_size], [num_layer * num_directions, batch, hidden_size])
        seq_len = memory_bank.shape[1]

        if self.use_topic_represent:                
            topic_represent_expand = topic_represent.unsqueeze(1).expand(self.batch_size, seq_len, self.topic_num)#.contiguous()    
            
            if self.attn_mode=='no_att':       
                logit = self.output_layer(memory_bank[:,-1,:])
                pred = F.log_softmax(logit, 1)    
                return pred  
                 
            if self.attn_mode=='concat':
                concat = torch.cat((memory_bank, topic_represent_expand), dim=-1)
                u = torch.tanh(self.W4(concat))
                g = self.W2(u)                                             
                alpha = F.softmax(g, 1)    
                context = memory_bank* alpha
                context = torch.sum(context, 1) 
                context = self.layer_norm(context)
                logit = self.output_layer(context)
                pred = F.log_softmax(logit, 1)    
                return pred
                                
            if self.attn_mode=='general':
                u = torch.tanh(self.W1(memory_bank)+ self.W3(topic_represent_expand))#length * hidden size
                g = self.W2(u)                       #length * 1                      
                alpha = F.softmax(g, 1)    
                context = memory_bank* alpha
                context = torch.sum(context, 1) 
                #context = self.layer_norm(context)
                logit = self.output_layer(context)
                pred = F.log_softmax(logit, 1)    
                return pred

            if self.attn_mode=='multipy':
                #cp(self.W3(topic_represent).shape,'self.W3(topic_represent)')
                #cp(memory_bank.shape,'memory_bank')
                u = torch.tanh(self.W3(topic_represent) @ memory_bank.transpose(1, 2)).transpose(1, 2)
                #cp(u.shape,'u')
                g = self.W2(u)                                             
                alpha = F.softmax(g, 1)    
                context = memory_bank* alpha
                context = torch.sum(context, 1) 
                #context = self.layer_norm(context)
                logit = self.output_layer(context)
                pred = F.log_softmax(logit, 1)    
                return pred
                      
            if self.attn_mode=='self_att':          
                u = torch.tanh(self.W1(memory_bank))
                g = self.W2(u)                                             
                alpha = F.softmax(g, 1)    
                context = memory_bank* alpha
                context = torch.sum(context, 1) 
                #context = self.layer_norm(context)
                logit = self.output_layer(context)
                pred = F.log_softmax(logit, 1)    
                return pred
                
            if self.attn_mode=='only_topic':       
                #context = self.layer_norm(context)
                logit = self.output_layer3(topic_represent)
                pred = F.log_softmax(logit, 1)    
                return pred  
                
            if self.attn_mode=='combine_no_att':
                u = torch.tanh(self.W1(memory_bank)+ self.W3(topic_represent_expand))
                g = self.W2(u)                                             
                alpha = F.softmax(g, 1)    
                context = memory_bank* alpha
                context = torch.sum(context, 1) 
                lstm_output = memory_bank[:,-1,:]
                concat = torch.cat((lstm_output, context), dim=-1)
                #context = self.layer_norm(context)
                logit = self.output_layer2(concat)
                pred = F.log_softmax(logit, 1)    
                return pred    
                             
            if self.attn_mode=='combine_self_att':
                u = torch.tanh(self.W1(memory_bank)+ self.W3(topic_represent_expand))
                g = self.W2(u)                                             
                alpha = F.softmax(g, 1)    
                context = memory_bank* alpha
                context = torch.sum(context, 1) 
                
                u1 = torch.tanh(self.W11(memory_bank))
                g1 = self.W22(u1) 
                alpha1 = F.softmax(g1, 1) 
                context1 = memory_bank* alpha1
                context1 = torch.sum(context1, 1)  
            
                #concat = torch.cat((context1, context), dim=-1)
                concat = context1+context
                #context = self.layer_norm(context)
                logit = self.output_layer(concat)
                pred = F.log_softmax(logit, 1)    
                return pred    
                
            if self.attn_mode=='combine_fusion':
                u = torch.tanh(self.W1(memory_bank)+ self.W3(topic_represent_expand))
                g = self.W2(u)                                             
                alpha = F.softmax(g, 1)    
                context = memory_bank* alpha
                context = torch.sum(context, 1) 
                lstm_output = memory_bank[:,-1,:]
                weight1 = torch.sigmoid(self.W23(context))  
                weight2 = torch.sigmoid(self.W24(lstm_output))
                weight1 = weight1/(weight1+weight2)
                weight2= 1-weight1
                
                concat = weight1*context+weight2*lstm_output
                #context = self.layer_norm(context)
                logit = self.output_layer(concat)
                pred = F.log_softmax(logit, 1)    
                return pred  
                                                                                
            if self.attn_mode=='MLA':  
                attention = self.attention(memory_bank).transpose(1, 2)
                weight = F.softmax(attention, -1)# N, labels_num, L
                pred = weight @ memory_bank # N, labels_num, hidden_size
                linear_out = pred
                for linear in self.linear:
                    linear_out = F.relu(linear(linear_out))
                logit = torch.squeeze(self.output(linear_out), -1)  
                pred = F.log_softmax(logit, 1) 
                return pred

            if self.attn_mode=='MTA': 
                topic_represent_expand = topic_represent.unsqueeze(1).expand(self.batch_size, self.hidden_size, self.topic_num)
                u = torch.tanh(memory_bank@self.W3(topic_represent_expand)) # N, seq_len, hidden size
                #topic_represent_expand: hidden_size * topic_num 
                #self.W3(topic_represent_expand):  hidden_size * hidden_size
                #memory_bank: len*hidden_size
                #u: len* hidden_size
                attention = self.attention(u).transpose(1, 2)# N, labels_num, L
                weight = F.softmax(attention, -1)# N, labels_num, L
                pred = weight @ memory_bank # N, labels_num, hidden_size
                linear_out = pred
                for linear in self.linear:# N, labels_num, linear1
                    linear_out = F.relu(linear(linear_out))# N, labels_num, linear2
                logit = torch.squeeze(self.output(linear_out), -1) # N, labels_num, 1
                pred = F.log_softmax(logit, 1) 
                return pred

            if self.attn_mode=='MTA2':  
                u = torch.tanh(self.W11(memory_bank)+ self.W3(topic_represent_expand))# N, seq_len, hidden size
                attention = self.attention(u).transpose(1, 2)# N, labels_num, L
                weight = F.softmax(attention, -1)# N, labels_num, L
                pred = weight @ memory_bank # N, labels_num, hidden_size
                linear_out = pred
                for linear in self.linear:# N, labels_num, linear1
                    linear_out = F.relu(linear(linear_out))# N, labels_num, linear2
                logit = torch.squeeze(self.output(linear_out), -1) # N, labels_num, 1
                pred = F.log_softmax(logit, 1) 
                return pred

            if self.attn_mode=='MTA3': 
                #topic_represent_expand = topic_represent.unsqueeze(1).expand(self.batch_size, self.hidden_size, self.topic_num)
                u = torch.tanh(self.W9(topic_represent_expand)@memory_bank) # N, seq_len, hidden size
                #topic_represent_expand: len * topic_num 
                #self.W9(topic_represent_expand):  len * len
                #memory_bank:   len*hidden_size
                #u: len* hidden_size
                attention = self.attention(u).transpose(1, 2)# N, labels_num, L
                weight = F.softmax(attention, -1)# N, labels_num, L
                pred = weight @ memory_bank # N, labels_num, hidden_size
                linear_out = pred
                for linear in self.linear:# N, labels_num, linear1
                    linear_out = F.relu(linear(linear_out))# N, labels_num, linear2
                logit = torch.squeeze(self.output(linear_out), -1) # N, labels_num, 1
                pred = F.log_softmax(logit, 1) 
                return pred
                
            if self.attn_mode=='MLA2':  
                attention = self.W6(memory_bank).transpose(1, 2)
                weight = F.softmax(attention, -1)# N, labels_num, L
                pred = weight @ memory_bank # N, labels_num, hidden_size
                linear_out = F.relu(self.W7(pred))
                logit = torch.squeeze(self.W8(linear_out), -1)  
                pred = F.log_softmax(logit, 1) 
                return pred
            
            if self.attn_mode=='MHA':  
                #mhatt = torch.tanh(self.W1(memory_bank))
                mhatt = self.W1(memory_bank)
                mhatt = self.W5(mhatt)
                mhatt = F.softmax(mhatt, dim=1)
                mhatt = mhatt.transpose(1, 2)
                context = torch.bmm(mhatt, memory_bank)
                context = torch.sum(context, 1) / self.att_dim  
                logit = self.output_layer(context)
                pred = F.log_softmax(logit, 1)    
                return pred                

            if self.attn_mode=='combine_MHA': 

                u = torch.tanh(self.W11(memory_bank)+ self.W3(topic_represent_expand))
                g = self.W2(u)                                             
                alpha = F.softmax(g, 1)    
                context = memory_bank* alpha
                context = torch.sum(context, 1) 

                mhatt = torch.tanh(self.W1(memory_bank))
                mhatt = self.W5(mhatt)
                mhatt = F.softmax(mhatt, dim=1)
                mhatt = mhatt.transpose(1, 2)
                context1 = torch.bmm(mhatt, memory_bank)
                context1 = torch.sum(context1, 1) / 3  
                
                concat = context1+context
                logit = self.output_layer(concat)
                pred = F.log_softmax(logit, 1)    
                return pred     

            if self.attn_mode=='combine_2MHA': 
                u = torch.tanh(self.W11(memory_bank)+ self.W3(topic_represent_expand))
                g = self.W51(u)                                             
                alpha = F.softmax(g, 1)  
                alpha = alpha.transpose(1, 2)  
                context = torch.bmm(alpha, memory_bank)
                context = torch.sum(context, 1) / self.att_dim                  

                mhatt = torch.tanh(self.W1(memory_bank))
                mhatt = self.W5(mhatt)
                mhatt = F.softmax(mhatt, dim=1)
                mhatt = mhatt.transpose(1, 2)
                context2 = torch.bmm(mhatt, memory_bank)
                context2 = torch.sum(context2, 1) / self.att_dim
                
                concat = context2+context
                #concat = self.layer_norm(concat)
                #doc = self.batch_norm(doc)
                logit = self.output_layer(concat)
                pred = F.log_softmax(logit, 1)    
                return pred  

            if self.attn_mode=='general_MHA':  
                u = torch.tanh(self.W1(memory_bank)+ self.W3(topic_represent_expand))
                g = self.W5(u)                                             
                alpha = F.softmax(g, 1)  
                alpha = alpha.transpose(1, 2)  
                context = torch.bmm(alpha, memory_bank)
                context = torch.sum(context, 1) / 3  
                logit = self.output_layer(context)
                pred = F.log_softmax(logit, 1)    
                return pred  
              
            if self.attn_mode=='LSAN':      
                u = torch.tanh(self.W3(topic_represent) @ memory_bank.transpose(1, 2)).transpose(1, 2)
                g = self.W51(u)                                             
                alpha = F.softmax(g, 1)  
                alpha = alpha.transpose(1, 2)  
                context = torch.bmm(alpha, memory_bank)
                #context = torch.sum(context, 1) / 30                  

                mhatt = torch.tanh(self.W1(memory_bank))
                mhatt = self.W5(mhatt)
                mhatt = F.softmax(mhatt, dim=1)
                mhatt = mhatt.transpose(1, 2)
                context2 = torch.bmm(mhatt, memory_bank)
                #context2 = torch.sum(context2, 1) / 30 
                
                weight1 = torch.sigmoid(self.W23(context))  
                weight2 = torch.sigmoid(self.W24(context2))
                weight1 = weight1/(weight1+weight2)
                weight2= 1-weight1
                
                doc = weight1*context+weight2*context2
                doc = torch.sum(doc, 1)/100
                #doc = self.layer_norm(doc)
                #doc = self.batch_norm(doc)
                logit = self.output_layer(doc)
                pred = F.log_softmax(logit, 1)    
                return pred                  
                         

            if self.attn_mode=='combine_2MHA_multipy': 
                u = torch.tanh(self.W3(topic_represent) @ memory_bank.transpose(1, 2)).transpose(1, 2)
                g = self.W51(u)                                             
                alpha = F.softmax(g, 1)  
                alpha = alpha.transpose(1, 2)  
                context = torch.bmm(alpha, memory_bank)
                context = torch.sum(context, 1) / self.att_dim                  

                mhatt = torch.tanh(self.W1(memory_bank))
                mhatt = self.W5(mhatt)
                mhatt = F.softmax(mhatt, dim=1)
                mhatt = mhatt.transpose(1, 2)
                context2 = torch.bmm(mhatt, memory_bank)
                context2 = torch.sum(context2, 1) / self.att_dim
                
                concat = context2+context
                #concat = self.layer_norm(concat)
                #doc = self.batch_norm(doc)
                logit = self.output_layer(concat)
                pred = F.log_softmax(logit, 1)    
                return pred  
                                                    
        if not self.use_topic_represent:
            #cp(memory_bank[:,-1,:].shape, 'memory_bank[:,-1,:].shape')
            #cp(self.hidden_size, 'self.hidden_size')
            logit = self.output_layer(memory_bank[:,-1,:])
            pred = F.log_softmax(logit, 1)    
            return pred  

class NTM(nn.Module):
    def __init__(self, opt, hidden_dim=500, l1_strength=0.001):
        super(NTM, self).__init__()
        
        self.input_dim = opt.bow_vocab
        self.topic_num = opt.topic_num
        topic_num = opt.topic_num        
        self.fc11 = nn.Linear(self.input_dim, hidden_dim)
        self.fc12 = nn.Linear(hidden_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, topic_num)
        self.fc22 = nn.Linear(hidden_dim, topic_num)
        self.fcs = nn.Linear(self.input_dim, hidden_dim, bias=False)
        self.fcg1 = nn.Linear(topic_num, topic_num)
        self.fcg2 = nn.Linear(topic_num, topic_num)
        self.fcg3 = nn.Linear(topic_num, topic_num)
        self.fcg4 = nn.Linear(topic_num, topic_num)
        self.fcd1 = nn.Linear(topic_num, self.input_dim)
        self.l1_strength = torch.FloatTensor([l1_strength]).to(opt.device)

    def encode(self, x):
        e1 = F.relu(self.fc11(x))
        e1 = F.relu(self.fc12(e1))
        e1 = e1.add(self.fcs(x))
        return self.fc21(e1), self.fc22(e1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu
            
    def generate(self, h):
        g1 = torch.tanh(self.fcg1(h))
        g1 = torch.tanh(self.fcg2(g1))
        g1 = torch.tanh(self.fcg3(g1))
        g1 = torch.tanh(self.fcg4(g1))
        g1 = g1.add(h)
        return g1
        
    def decode(self, z):
        d1 = F.softmax(self.fcd1(z), dim=1)
        return d1

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.input_dim))
        z = self.reparameterize(mu, logvar)
        g = self.generate(z)
        return z, g, self.decode(g), mu, logvar

    def print_topic_words(self, vocab_dic, fn, n_top_words=10):
        beta_exp = self.fcd1.weight.data.cpu().numpy().T
        logging.info("Writing to %s" % fn)
        fw = open(fn, 'w')
        for k, beta_k in enumerate(beta_exp):
            topic_words = [vocab_dic[w_id] for w_id in np.argsort(beta_k)[:-n_top_words - 1:-1]]
            print('Topic {}: {}'.format(k, ' '.join(topic_words)))
            fw.write('{}\n'.format(' '.join(topic_words)))
        fw.close()
            

def myModelStat(model):
    print('===========================Model Para==================================')
    # Find total parameters and trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')
    for name,parameters in model.named_parameters():
        print(name,':',parameters.size())     
    print('===========================Model Para==================================')

'''
        if self.title_aware:
            title = memory_bank[:,:self.title_len,:]
            body = memory_bank[:,self.title_len:,:]
            temp = self.W_title(body)
            #cp(temp.shape,'temp.shape')
            #cp(title.shape,'title.shape')
            s = torch.bmm(temp, title.permute(0,2,1))#batch_size*body_len*title_len
            alpha = F.softmax(s, 1)    
            c = torch.bmm(alpha, title)#batch_size*body_len*hidden_size
                
            memory_bank2 = torch.cat((body, c), dim=-1)#batch_size*body_len*2hidden_size
            memory_bank3, encoder_final_state = self.rnn2(memory_bank2)
            memory_bank4 = self.lambdaa*body + (1-self.lambdaa)*memory_bank3
            memory_bank = memory_bank4
            seq_len = memory_bank.shape[1]
            #cp(memory_bank4.shape,'memory_bank4.shape')

            uu = torch.tanh(self.W11(memory_bank4))
            gg = self.W22(uu)
            alpha2 = F.softmax(gg, 1)
            #cp(memory_bank4.shape,'memory_bank4.shape')
            #cp(alpha2.shape,'alpha2.shape')
            
            context2 = torch.bmm(memory_bank4.permute(0,2,1), alpha2)
            context2 = torch.squeeze(context2)
            #cp(context2.shape,'context2.shape')    
            #context = self.layer_norm(context)
            logit = self.output_layer(context2)
            pred = F.log_softmax(logit, 1)    
            return pred      
       
'''
'''
        self.lambdaa = opt.lambdaa
        self.title_aware = opt.title_aware
        self.title_ratio = opt.title_ratio  
        self.max_src_len = opt.max_src_len   
        self.title_len = int(self.title_ratio*self.max_src_len)
        self.body_len = self.max_src_len-self.title_len
        self.W_title = torch.nn.Linear(self.hidden_size, self.hidden_size)
       self.rnn2 = torch.nn.GRU(input_size=self.hidden_size*2, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True, bidirectional=self.bidirectional, dropout=self.drop_rate)    
'''      
'''
                    if self.attn_mode=='title_att':
                
                u = torch.tanh(self.W1(memory_bank)+ self.W3(topic_represent_expand))
                g = self.W2(u)                                             
                alpha = F.softmax(g, 1)    
                context = memory_bank* alpha
                context = torch.sum(context, 1) #batch_size*hidden_size
                context = context.unsqueeze(1).expand(self.batch_size, seq_len, self.hidden_size)#batch_size*seq_len*hidden_size
                
                memory_bank2 = torch.cat((memory_bank, context), dim=-1)#batch_size*seq_len*2hidden_size
                memory_bank3, encoder_final_state = self.rnn2(memory_bank2)
                memory_bank4 = self.lambdaa*memory_bank + (1-self.lambdaa)*memory_bank3
                
                uu = torch.tanh(self.W11(memory_bank4))
                gg = self.W22(uu)
                alpha2 = F.softmax(gg, 1)
                context2 = memory_bank4* alpha2
                context2 = torch.sum(context2, 1) 
                
                #context = self.layer_norm(context)
                logit = self.output_layer(context2)
                pred = F.log_softmax(logit, 1)    
                return pred                    
'''               