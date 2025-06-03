params = {'solo_weakUK': { 'drop': 0.1,
                         'kernel': 5,
                         'layers': 4,
                          'GRU': 16,
                           'cs':True,
                           'no_weak' : False,
                           'pre_trained': '/home/eprincipi/Weak_Supervision/weak_transfer_learning/CRNN_model_NOISED_BEST_100weak_0strong_LINSOFTMAX.h5'},
          'strong_weakUK': { 'drop': 0.1,
                         'kernel': 5,
                         'layers': 3,
                          'GRU': 64,
                          'cs': False,
                            'no_weak' : False,
                          'pre_trained': '/raid/users/eprincipi/Knowledge_Distillation/pretrained/CRNN_model_PRETRAINED_ukdale_SWW_correct0.01.h5'}, #
          'solo_strongUK': {'drop': 0.1,
                            'kernel': 5,
                            'layers': 3,
                            'GRU': 64,
                            'cs': False,
                            'no_weak' : True,
                            'pre_trained': '/home/eprincipi/Weak_Supervision/weak_transfer_learning/CRNN_model_NOISED_BEST_only_100strong_LINSOFTMAX.h5'},

          'mixed' : { 'drop': 0.1,
                         'kernel': 5,
                         'layers': 3,
                          'GRU': 64,
                        'cs': False,
                        'no_weak' : False,
                        'pre_trained': '/home/eprincipi/Weak_Supervision/weak_transfer_learning/CRNN_model_UKDALE_REFIT_RESAMPLED_60weak_20strong_weak_fixed_seed0.001.h5'},
          'strong_weakREFIT': { 'drop': 0.1,
                             'kernel': 5,
                             'layers': 3,
                              'GRU': 64,
                              'cs': False,
                              'no_weak' : False,
                              'pre_trained': 'src/teacher_model/teacher_strong_weakREFIT_FocalLoss_new221_64.h5'
              }}



uk_params = {'mean': 453.57,
            'std': 729.049}
refit_params = {'mean': 537.051,
                'std': 746.905}