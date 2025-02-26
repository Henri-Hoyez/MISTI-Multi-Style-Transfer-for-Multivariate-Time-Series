from models.backbone import StyleEncoder, GlobalDiscriminator, LocalDiscriminator, ContentEncoder, Decoder

from configs.MetricConfig import Metric



def get_dataset_class(dataset_name):
    """Return the algorithm class with the given name."""
    # Stolen from the RainCOAT repository :P
    # Thanks !
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]


##       ###       ##
# SYNTHETIC DATASET #
##       ###       ##
class Synthetic():
    def __init__(self):
        self.sequence_lenght_in_sample = 64
        self.n_feature = 7
        
        self.granularity = 1
        self.overlap= 0.05
        
        self.n_classes = 5
        self.n_styles= 10
        self.batch_size = 256
        
        self.style_vector_size = 16
        self.learning_rate = 0.0002
        
        self.l_global_reconstr = 0.3
        self.l_local_reconstr = 1- self.l_global_reconstr
        
        self.n_wiener = 2
        self.n_sample_wiener = self.sequence_lenght_in_sample//8
        self.unsupervised = False
        self.drop_labels= False
        self.univariate = False
        
        self.content_dataset = "data/simulated_dataset/01 - Source Domain_standardized.h5"
        
        self.met_params = Metric()
        
    def get_generator(self):
        return Decoder.make_generator(self.n_sample_wiener, self.n_wiener, self.style_vector_size, self.n_feature, [16, 32, 64])
    
    def get_content_encoder(self):
        return ContentEncoder.content_encoder(self.sequence_lenght_in_sample, self.n_feature, [16, 32, 64])
    
    def get_style_encoder(self):
        return StyleEncoder.make_style_encoder(self.sequence_lenght_in_sample, 
                                                self.n_feature,
                                                self.style_vector_size,
                                                [16, 32, 64, 128])
    def get_global_discriminator(self):
        return GlobalDiscriminator.make_global_discriminator(self.sequence_lenght_in_sample,
                                                             self.n_feature, 
                                                             self.n_styles, 
                                                             [32, 64], dropout=0.1)
    
    def get_local_discriminator(self):
        return LocalDiscriminator.create_local_discriminator(self.n_feature,
                                                            self.sequence_lenght_in_sample, 
                                                            [8, 16], dropout=0.1)



class InputNoise(Synthetic):
    def __init__(self) -> None:
        super(InputNoise, self).__init__()
        
        ##### Generator loss parameters.
        self.l_reconstr = 6.0
        self.l_local =  1.5
        self.l_global = 1.5
        self.l_style_preservation = 1.5
        
        self.l_global_reconstr = 0.3
        self.l_local_reconstr = 1- self.l_global_reconstr

        ##### Content encoder loss
        self.l_content =1.5
        self.encoder_adv = 0.05
        self.encoder_recontr = 2.
        

        ##### Style Encoder
        self.l_disentanglement = 1.5
        self.l_triplet = 1.5
        self.triplet_r = 0.05
                
        self.style_datasets = [
            "data/simulated_dataset/input_noise/0.25_standardized.h5", 
            "data/simulated_dataset/input_noise/0.50_standardized.h5",
            "data/simulated_dataset/input_noise/0.75_standardized.h5", 
            "data/simulated_dataset/input_noise/1.00_standardized.h5",
            "data/simulated_dataset/input_noise/1.25_standardized.h5", 
            "data/simulated_dataset/input_noise/1.50_standardized.h5",
            "data/simulated_dataset/input_noise/1.75_standardized.h5", 
            "data/simulated_dataset/input_noise/2.00_standardized.h5",
            "data/simulated_dataset/input_noise/2.25_standardized.h5", 
            "data/simulated_dataset/input_noise/2.50_standardized.h5",
        ]
    
    # def get_generator(self):
    #     return Decoder.make_generator(self.n_sample_wiener, self.n_wiener, self.style_vector_size, self.n_feature, [32, 64, 128])
    
    # def get_content_encoder(self):
    #     return ContentEncoder.content_encoder(self.sequence_lenght_in_sample, self.n_feature, [32, 64, 128])

    # def get_style_encoder(self):
    #     return StyleEncoder.make_style_encoder(self.sequence_lenght_in_sample, 
    #                                             self.n_feature,
    #                                             self.style_vector_size,
    #                                             [8, 16, 32, 64])
    
    # def get_local_discriminator(self):
    #     return LocalDiscriminator.create_local_discriminator(self.n_feature,
    #                                                         self.sequence_lenght_in_sample, 
    #                                                         [16, 32], dropout=0.25)
    
    # def get_global_discriminator(self):
    #     return GlobalDiscriminator.make_global_discriminator(self.sequence_lenght_in_sample,
    #                                                          self.n_feature, 
    #                                                          self.n_styles, 
    #                                                          [32, 64], dropout=0.25)

class OutputNoise(Synthetic):
    
    def __init__(self) -> None:
        super(OutputNoise, self).__init__()
        ##### Generator loss parameters.
        self.l_reconstr = 6.0
        self.l_local =  0.8
        self.l_global = 0.8
        self.l_style_preservation = 1.5
        
        self.l_global_reconstr = 0.3
        self.l_local_reconstr = 1- self.l_global_reconstr

        ##### Content encoder loss
        self.l_content =1.5
        self.encoder_adv = 0.05
        self.encoder_recontr = 2.
        

        ##### Style Encoder
        self.l_disentanglement = 1.5
        self.l_triplet = 1.5
        self.triplet_r = 0.05
        
        
        self.style_datasets = [
            "data/simulated_dataset/output_noise/0.25_standardized.h5", 
            "data/simulated_dataset/output_noise/0.50_standardized.h5",
            "data/simulated_dataset/output_noise/0.75_standardized.h5", 
            "data/simulated_dataset/output_noise/1.00_standardized.h5",
            "data/simulated_dataset/output_noise/1.25_standardized.h5", 
            "data/simulated_dataset/output_noise/1.50_standardized.h5",
            "data/simulated_dataset/output_noise/1.75_standardized.h5", 
            "data/simulated_dataset/output_noise/2.00_standardized.h5",
            "data/simulated_dataset/output_noise/2.25_standardized.h5", 
            "data/simulated_dataset/output_noise/2.50_standardized.h5",
        ]
        
    # def get_generator(self):
    #     return Decoder.make_generator(self.n_sample_wiener, self.n_wiener, self.style_vector_size, self.n_feature, [8, 16, 32])
    
    # def get_content_encoder(self):
    #     return ContentEncoder.content_encoder(self.sequence_lenght_in_sample, self.n_feature, [8, 16, 32])

    # def get_style_encoder(self):
    #     return StyleEncoder.make_style_encoder(self.sequence_lenght_in_sample, 
    #                                             self.n_feature,
    #                                             self.style_vector_size,
    #                                             [16, 32, 64, 128])
    
    # def get_local_discriminator(self):
    #     return LocalDiscriminator.create_local_discriminator(self.n_feature,
    #                                                         self.sequence_lenght_in_sample, 
    #                                                         [16, 16], dropout=0.1)
    
    # def get_global_discriminator(self):
    #     return GlobalDiscriminator.make_global_discriminator(self.sequence_lenght_in_sample,
    #                                                          self.n_feature, 
    #                                                          self.n_styles, 
    #                                                          [16, 32], dropout=0.1)

class TimeShift(Synthetic):
    def __init__(self) -> None:
        super(TimeShift, self).__init__()
        
        ##### Generator loss parameters.
        self.l_reconstr = 6.0
        self.l_local =  1.5
        self.l_global = 1.5
        self.l_style_preservation = 1.5
        
        self.l_global_reconstr = 0.3
        self.l_local_reconstr = 1- self.l_global_reconstr

        ##### Content encoder loss
        self.l_content =1.5
        self.encoder_adv = 0.05
        self.encoder_recontr = 2.
        

        ##### Style Encoder
        self.l_disentanglement = 1.5
        self.l_triplet = 1.5
        self.triplet_r = 0.05
        
        self.style_datasets = [
            "data/simulated_dataset/time_shift/0_standardized.h5",
            "data/simulated_dataset/time_shift/2_standardized.h5",
            "data/simulated_dataset/time_shift/4_standardized.h5",
            "data/simulated_dataset/time_shift/6_standardized.h5",
            "data/simulated_dataset/time_shift/8_standardized.h5",
            "data/simulated_dataset/time_shift/10_standardized.h5",
            "data/simulated_dataset/time_shift/12_standardized.h5",
            "data/simulated_dataset/time_shift/14_standardized.h5",
            "data/simulated_dataset/time_shift/16_standardized.h5",
            "data/simulated_dataset/time_shift/18_standardized.h5"
        ]
    
    # def get_local_discriminator(self):
    #     return LocalDiscriminator.create_local_discriminator(self.n_feature,
    #                                                         self.sequence_lenght_in_sample, 
    #                                                         [32, 32], dropout=0.25)
    
    # def get_global_discriminator(self):
    #     return GlobalDiscriminator.make_global_discriminator(self.sequence_lenght_in_sample,
    #                                                          self.n_feature, 
    #                                                          self.n_styles, 
    #                                                          [32, 64], dropout=0.25)



class CausalShift(Synthetic):
    def __init__(self) -> None:
        super(CausalShift, self).__init__()
        
        ##### Generator loss parameters.
        self.l_reconstr = 6.0
        self.l_local =  1.5
        self.l_global = 1.5
        self.l_style_preservation = 1.5
        
        self.l_global_reconstr = 0.3
        self.l_local_reconstr = 1- self.l_global_reconstr

        ##### Content encoder loss
        self.l_content =1.5
        self.encoder_adv = 0.05
        self.encoder_recontr = 2.
        

        ##### Style Encoder
        self.l_disentanglement = 1.5
        self.l_triplet = 1.5
        self.triplet_r = 0.05
        
        self.style_datasets = [
            "data/simulated_dataset/causal_shift/0.00_standardized.h5", 
            "data/simulated_dataset/causal_shift/0.10_standardized.h5", 
            "data/simulated_dataset/causal_shift/0.20_standardized.h5", 
            "data/simulated_dataset/causal_shift/0.30_standardized.h5", 
            "data/simulated_dataset/causal_shift/0.40_standardized.h5", 
            "data/simulated_dataset/causal_shift/0.50_standardized.h5", 
            "data/simulated_dataset/causal_shift/0.60_standardized.h5", 
            "data/simulated_dataset/causal_shift/0.70_standardized.h5", 
            "data/simulated_dataset/causal_shift/0.80_standardized.h5", 
            "data/simulated_dataset/causal_shift/0.90_standardized.h5"
        ]

##       ###       ##
#    HAR DATASET    #
##       ###       ##
class HAR():
    def __init__(self) -> None:
        ####### DATA LOADING PARAMETERS
        self.unsupervised = False
        self.drop_labels= False
        self.univariate = False
        
        self.sequence_lenght_in_sample = 128
        self.granularity = 2
        self.n_feature = 6
        
        self.overlap= 0.02
        
        self.n_classes = 5
        self.n_styles = 3
        
        self.style_vector_size = 16
        
        self.learning_rate = 0.002       
        self.batch_size = 64
        
        self.n_wiener = 2
        self.n_sample_wiener = self.sequence_lenght_in_sample//8
        print("sample wiener:", self.n_sample_wiener)
        
        ##### Generator loss parameters.
        self.l_reconstr = 2.
        self.l_local =  6.
        self.l_global = 6.
        self.l_style_preservation = 1.5
        
        ## Local V.s. local reconstruction loss
        self.l_global_reconstr = 0.3
        self.l_local_reconstr = 1- self.l_global_reconstr

        ##### Content encoder loss
        self.l_content = 2.
        self.encoder_adv = 0.1
        self.encoder_recontr = 2.
        

        ##### Style Encoder
        self.l_disentanglement = 4.
        self.l_triplet = 4. # 1.5
        self.triplet_r = 0.01
        
        self.content_dataset = "data/PAMAP2/subject101.h5"
        
        self.met_params = Metric()
        self.met_params.signature_length= self.sequence_lenght_in_sample//2
        
        self.style_datasets = [
            "data/PAMAP2/subject105.h5",
            "data/PAMAP2/subject106.h5",
            "data/PAMAP2/subject108.h5"
        ]
        
    def get_content_encoder(self):
        return ContentEncoder.content_encoder(self.sequence_lenght_in_sample,
                                              self.n_feature, 
                                              [16, 32, 64])
        
    def get_generator(self):
        return Decoder.make_generator(self.n_sample_wiener, self.n_wiener, self.style_vector_size, self.n_feature, 
                                      [16, 32, 64])
    
    def get_style_encoder(self):
        return StyleEncoder.make_style_encoder(self.sequence_lenght_in_sample, 
                                                self.n_feature,
                                                self.style_vector_size,
                                                [16, 32, 64])
    
    
    def get_local_discriminator(self):
        return LocalDiscriminator.create_local_discriminator(self.n_feature,
                                                            self.sequence_lenght_in_sample, [16, 16, 32], dropout=0.)
    
    def get_global_discriminator(self):
        return GlobalDiscriminator.make_global_discriminator(self.sequence_lenght_in_sample,
                                                            self.n_feature, 
                                                            self.n_styles, [16, 32, 64], dropout=0.)
    
##            ###        ##
#       GOOGLE STOCKS     #
##            ###        ##
class UnivariateDatasets():
    def __init__(self) -> None:
        self.unsupervised = True
        self.drop_labels= False
        self.univariate = True
        
        self.sequence_lenght_in_sample = 32
        self.n_feature = 2 # Same sequence both time to adapt the univariate to multivariate.   
        
        self.granularity = 1
        self.overlap= 0.05
        
        self.n_classes = 2 # Will disable the Classification metric
        self.batch_size = 16
        
        self.style_vector_size = 8
    
        self.n_wiener = 2
        self.n_sample_wiener = self.sequence_lenght_in_sample//4
        
        self.learning_rate = 0.0002
        
        ##### Generator loss parameters.
        self.l_reconstr = 6.0
        self.l_local =  1.5
        self.l_global = 1.5
        self.l_style_preservation = 1.5
        
        self.l_global_reconstr = 0.3
        self.l_local_reconstr = 1- self.l_global_reconstr

        ##### Content encoder loss
        self.l_content =1.5
        self.encoder_adv = 0.05

        ##### Style Encoder
        self.l_disentanglement = 1.5
        self.l_triplet = 1.5
        self.triplet_r = 0.05
        
    def get_generator(self):
        return Decoder.make_generator(self.n_sample_wiener, self.n_wiener, self.style_vector_size, self.n_feature, [16, 16])

    def get_content_encoder(self):
        return ContentEncoder.content_encoder(self.sequence_lenght_in_sample, self.n_feature, [8, 16])
    

    def get_style_encoder(self):
        return StyleEncoder.make_style_encoder(self.sequence_lenght_in_sample, 
                                                          self.n_feature,
                                                          self.style_vector_size, [16, 32])
    
    def get_local_discriminator(self):
        return LocalDiscriminator.create_local_discriminator(self.n_feature,
                                                                 self.sequence_lenght_in_sample, 
                                                                 [8, 16])
    
    def get_global_discriminator(self):
        return GlobalDiscriminator.make_global_discriminator(self.sequence_lenght_in_sample,
                                                                        self.n_feature, 
                                                                        self.n_classes,
                                                                        [8, 8])
    
class GoogleStocksInSample(UnivariateDatasets):
    def __init__(self) -> None:
        super(GoogleStocksInSample, self).__init__()
        
        self.content_dataset = "data/google_stocks/in_sample.npy"
        
        self.style_datasets = [
            "data/google_stocks/style.npy",
            "data/google_stocks/perturbed.npy"
        ]
    
    
class GoogleStocksPerturbed(UnivariateDatasets):
    def __init__(self) -> None:
        super(GoogleStocksPerturbed, self).__init__()
        self.content_dataset = "data/google_stocks/perturbed.npy"
        
        self.style_datasets = [
            "data/google_stocks/style.npy",
            "data/google_stocks/in_sample.npy"
        ]
        
    
    
##       ###        ##
#       Energy       #
##       ###        ##
class EnergyAppliancesInSample(UnivariateDatasets):
    def __init__(self) -> None:
        super(EnergyAppliancesInSample, self).__init__()
        
        self.content_dataset = "data/energy/in_sample.npy"
        
        self.style_datasets = [
            "data/energy/style.npy",
            "data/energy/perturbed.npy"
        ]
    
    
class EnergyAppliancesPerturbed(UnivariateDatasets):
    def __init__(self) -> None:
        super(EnergyAppliancesPerturbed, self).__init__()
        
        self.content_dataset = "data/energy/perturbed.npy"
        
        self.style_datasets = [
            "data/energy/style.npy",
            "data/energy/in_sample.npy"
        ]