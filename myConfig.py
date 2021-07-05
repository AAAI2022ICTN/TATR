import logging
import os
import sys
import time

def init_logging(log_file, stdout=False):
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(module)s: %(message)s',
                                  datefmt='%m/%d/%Y %H:%M:%S')

    print('Making log output file: %s' % log_file)
    print(log_file[: log_file.rfind(os.sep)])
    if not os.path.exists(log_file[: log_file.rfind(os.sep)]):
        os.makedirs(log_file[: log_file.rfind(os.sep)])

    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    fh.setLevel(logging.INFO)

    logger = logging.getLogger()
    logger.addHandler(fh)
    logger.setLevel(logging.INFO)

    if stdout:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        ch.setLevel(logging.INFO)
        logger.addHandler(ch)

    return logger

def ntm_opts(parser):
    #joint leanring
    parser.add_argument('-joint_train', default=False, action='store_true')
    parser.add_argument('-joint_train_strategy', default='p_1_joint', choices=['p_1_iterate', 'p_0_iterate', 'p_1_joint', 'p_0_joint'])
    parser.add_argument('-iterate_train_ntm', default=False, action='store_true')#?
    parser.add_argument('-p_fe_e', type=int, default=3, help='number of epochs for training fe before joint train')
    parser.add_argument('-add_two_loss', default=False, action='store_true')#?
    parser.add_argument('-save_each_epoch', default=False, action='store_true')
    parser.add_argument('-start_checkpoint_at', type=int, default=2,
                        help="""Start checkpointing every epoch after and including
                        this epoch""")           
    parser.add_argument('-delta', type=float, default=0.1, help="Target sparsity for ntm model")
    
    #only train ntm
    parser.add_argument('-load_pretrain_ntm', default=False, action='store_true')
    parser.add_argument('-only_train_ntm', default=False, action='store_true')
    parser.add_argument('-check_pt_ntm_model_path', type=str)
    parser.add_argument('-ntm_warm_up_epochs', type=int, default=100)
           
    # different topic configurations
    parser.add_argument('-use_topic_represent', default=False, action='store_true', help="Use topic represent in the fe")
    parser.add_argument('-topic_num', type=int, default=300)
    parser.add_argument('-topic_type', default='z', choices=['z', 'g'], help='use latent variable z or g as topic')
    parser.add_argument('-attn_mode', type=str, default='MHA',\
    choices=['general', 'concat','combine_no_att','combine_fusion','MLA','only_topic','no_att',\
    'self_att','combine_self_att','MHA','combine_MHA','general_MHA','combine_2MHA','LSAN','multipy','combine_2MHA_multipy','MLA2','MTA','MTA2','MTA3'], help="""The attention type to use""")
    parser.add_argument('-target_sparsity', type=float, default=0.7, help="Target sparsity for ntm model")
                       
def model_opts(parser):

    # GPU
    parser.add_argument('-gpuid', default=5, type=int,help="Use CUDA on the selected device.")  
    # MLAttention
    parser.add_argument('-linear_size', default=256, type=int,help="linear_size of MLAttention.") 
    parser.add_argument('-att_dim', default=300, type=int,help="Dim of Attention.") 
    
    #Data
    parser.add_argument('-load_path', default='/data/pengyu/tag_rec/AU/' ,type=str,metavar='PATH', help='The dataset we use')
    parser.add_argument('-vocab_size', type=int, default=50000, help="Size of the vocab dictionary")      
    parser.add_argument('-bow_vocab', type=int, default=10000, help="Size of the bow dictionary") 
    parser.add_argument('-max_src_len', type=int, default=200, help="length of documnet") 
    parser.add_argument('-max_trg_len', type=int, default=5, help="length of tag number")      
                
    #model
    parser.add_argument('-num_layers', default=1, type=int,help="num_layers")  
    parser.add_argument('-emb_size', default=100, type=int,help="emb_size")    
    parser.add_argument('-hidden_size', default=128, type=int, help="lstm_hidden_dimension")  
    parser.add_argument('-drop_rate', default=0.1, type=int,help="drop_rate")   
    parser.add_argument('-bidirectional', default=True, action="store_true", help="whether the encoder is bidirectional")   
             
    #Train
    parser.add_argument('-seed', type=int, default=9527, help="""Random seed used for the experiments reproducibility.""")
    parser.add_argument('-batch_size', default=256, type=int, help="batch_size") 
    parser.add_argument('-learning_rate', default=0.001, type=int,help="learning_rate")  
    parser.add_argument('-epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('-start_epoch', type=int, default=1, help='The epoch from which to start')   
  
    # Path
    timemark = time.strftime('%Y%m%d-%H%M%S', time.localtime(time.time()))
    parser.add_argument('-timemark', type=str, default=timemark, help="The current time stamp.")
    parser.add_argument('-model_path', type=str, default="model/%s.%s", help="Path of checkpoints.") 
    parser.add_argument('-exp', type=str, default="AU", help="Name of the experiment for logging.")

    #debug
    parser.add_argument('-sample_data', default=False, action='store_true') 
    parser.add_argument('-sample_data_size', type=int, default=.07, help='number of epochs for training fe before joint train')  

    #early stopping
    parser.add_argument('-learning_rate_decay', type=float, default=0.5,
                        help="""If update_learning_rate, decay learning rate by
                        this much if (i) perplexity does not decrease on the
                        validation set or (ii) epoch has gone past
                        start_decay_at""")
    parser.add_argument('-early_stop_tolerance', type=int, default=1,
                        help="Stop training if it doesn't improve any more for several rounds of validation")                
        
    parser.add_argument('-train_from', default='', type=str,
                        help="""If training from a checkpoint then this is the
                        path to the pretrained model's state_dict.""")  
    '''
    #title
    parser.add_argument('-title_aware', default=False, action='store_true')   
    parser.add_argument('-title_ratio',type=float, default=0.01, help="Size of the vocab dictionary") 
    parser.add_argument('-lambdaa',type=float, default=0.9, help="merge layer")
    '''              