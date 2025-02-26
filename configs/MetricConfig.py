class Metric():
    def __init__(self) -> None:
        super().__init__()

        self.signature_length = 32
        self.sequence_to_generate = 500

        self.ins = [0, 1]
        self.outs =[2, 3, 4, 5]
        self.mean_senssibility_factor = 0
        self.noise_senssitivity = 0.5 
        
class Proposed:
    sampling_period = 5 # Sampling period in minutes
    smoothing_period = 1*60 # Sampling period in minutes
    cols_on_interrest = ['in_c1', 'in_c2', 'out_c1', 'out_c2', 'out_c3', 'out_c4']
    
    use_mean_scaling = False

    sequence_lenght_in_sample = 64
    granularity = 1
    overlap= 0.05
    epochs = 40

    n_feature = 7
    seq_shape = (sequence_lenght_in_sample, n_feature)
    batch_size = 64

    train_split = 0.7
    test_split  = 0.3
    valid_split = 0.2 

    reduce_train_set = False
    valid_set_batch_size= 50

    # loss Parameters:
    n_styles = 2
    style_vector_size = 16
    n_wiener = 2
    n_sample_wiener = 8 #sequence_lenght_in_sample//4
    noise_dim = (n_sample_wiener, n_wiener)
    n_validation_sequences = 500
    discrinator_step = 1

    ##### Generator loss parameters.
    l_reconstr = 2
    l_local =  1.5#.2
    l_global = 1.5
    l_style_preservation = 1.

    ##### Content encoder loss
    l_content = 1.5
    encoder_adv = 0.1

    ##### Style Encoder
    l_disentanglement = 1.5
    l_triplet = 1.5
    triplet_r = 0.01

    # Train the generator or the discriminator based on the 
    # Performance of the Discriminator (Here the accuracy.)
    discriminator_success_threashold = 0.75
    alpha = 0.01
    normal_training_epochs = 0

    met_params = Metric()
        

