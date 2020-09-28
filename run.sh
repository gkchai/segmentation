#!/usr/bin/env bash

python main.py --phase train --exp "unet_tones" --train_data 'isic17' 'isic18' 'usr_mob' 'usr_d415' 'tones' --test_data 'isic17' 'usr_mob' 'tones' --nb_epochs 250
python main.py --phase train --exp "unet_tones_entropyloss" --loss sce --train_data 'isic17' 'isic18' 'usr_mob' 'usr_d415' 'tones' --test_data 'isic17' 'usr_mob' 'tones' --nb_epochs 250

python main.py --phase train --exp "unet_notones" --nb_epochs 250

python main.py --phase train --exp "unet_notones_entropyloss" --loss sce --nb_epochs 250

python main.py --phase train --exp "unet_dilated_notones_" --model unet_dilated --nb_epochs 250

python main.py --phase train --exp "unet_dilated_tones" --model unet_dilated --train_data 'isic17' 'isic18' 'usr_mob' 'usr_d415' 'tones' --test_data 'isic17' 'usr_mob' 'tones' --nb_epochs 250

python main.py --phase train --exp "256_no_tones"  --train_data 'isic17' 'isic18' 'usr_mob' 'usr_d415' --test_data 'isic17' 'usr_mob' --dim 256 --nb_epochs 250