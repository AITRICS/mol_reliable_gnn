class Args():
    def __init__(self, config):
        self.config = config
        self.gpu_id = None 
        self.model = None
        self.dataset = None
        self.data_seed = None
        self.num_train = None
        self.num_val = None
        self.num_test = None
        self.out_dir = None
        self.seed = None
        self.epochs = None
        self.batch_size = None
        self.init_lr = None
        self.lr_reduce_factor = None
        self.lr_schedule_patience = None
        self.min_lr = None
        self.weight_decay = None
        self.print_epoch_interval = None
        self.L = None
        self.hidden_dim = None
        self.out_dim = None
        self.residual = None
        self.edge_feat = None
        self.readout = None
        self.n_heads = None
        self.gated = None
        self.in_feat_dropout = None
        self.dropout = None
        self.graph_norm = None
        self.batch_norm = None
        self.embedding_dim = None
        self.self_loop = None
        self.max_time = None
        self.atom_meta = None
        self.scaffold_split = None
        self.scheduler = None
        self.step_size = None
        self.step_gamma = None
        self.layer_norm = None

        self.optimizer = 'ADAM'
        self.grad_clip = 0.
        
        #   Additional arguments for GCN
        self.agg = None
        
        #   Additional arguments for GAT
        self.att_reduce_fn = None
        
        #   Additional arguments for SAGE 
        self.sage_aggregator = None
        self.concat_norm = None
        
        #   Additional arguments for GIN
        self.neighbor_aggr_GIN = None
        
        #   Additional arguments for GatedGCN
        self.gated_gcn_agg = None
        
        #   additional arguments for pretraining
        self.save_params = None
        self.pretrain = None
        self.input_model_file = None
        self.output_model_file = None

        #   additional arguments for mcdropout training
        self.mcdropout = False
        self.mc_eval_num_samples = 30

        #   additional arguments for SWA/SWAG training
        self.swa = False
        self.swag = False
        self.swa_start = 150
        self.swa_lr_alpha1 = 0.01
        self.swa_lr_alpha2 = 0.001
        self.swa_c_epochs = 4
        self.swag_eval_scale = 1.
        self.swag_eval_num_samples = 30

        #   additional arguments for SGLD training
        self.sgld = False
        self.psgld = False
        self.sgld_noise_std = 0.001
        self.sgld_save_every = 2
        self.sgld_save_start = 100
        self.sgld_max_samples = 100

        #   additional arguments for BBP training
        self.bbp = False
        self.bbb_prior_sigma_1 = 0.1
        self.bbb_prior_sigma_2 = 0.001
        self.bbb_prior_pi = 1.0
        self.bbp_complexity = 0.1
        self.bbp_sample_nbr = 5
        self.bbp_eval_Nsample = 100

