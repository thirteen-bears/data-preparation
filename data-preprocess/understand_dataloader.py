import argparse
from easydict import EasyDict
from dataloader import AudiosetDataset

args = {}
args['data_train'] = './sample_datafiles/sample_json_subset.json'
args['label_csv'] = './sample_datafiles/class_labels_indices_subset.csv'
args['roll_mag_aug'] = False #use roll_mag_aug

# for audio_conf 
args['freqm']  = 0  # frequency mask max length, pretraining 0
args['timem'] = 0  # time mask max length, pretraining 0
args['mixup'] = 0 # how many (0-1) samples need to be mixup during training
args['dataset'] = "audioset"  # choices=["audioset", "esc50", "speechcommands"]
args['load_video'] = False
args = EasyDict(args)

target_length = {'audioset':1024, 'esc50':512, 'speechcommands':128}
norm_stats = {'audioset':[-4.2677393, 4.5689974], 'esc50':[-6.6268077, 5.358466], 'speechcommands':[-6.845978, 5.5654526]}
multilabel_dataset = {'audioset': True, 'esc50': False, 'k400': False, 'speechcommands': True}
audio_conf = {'num_mel_bins': 128, 
              'target_length': target_length[args.dataset], 
              'freqm': args.freqm,
              'timem': args.timem,
              'mixup': args.mixup,
              'dataset': args.dataset,
              'mode':'train',
              'mean':norm_stats[args.dataset][0],
              'std':norm_stats[args.dataset][1],
              'multilabel':multilabel_dataset[args.dataset],
              'noise':False}

dataset_train = AudiosetDataset(args.data_train, label_csv=args.label_csv, audio_conf=audio_conf, 
                                #roll_mag_aug=args.roll_mag_aug,
                                #load_video=args.load_video
                               )



temp = iter(dataset_train)
next(temp)