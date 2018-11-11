# Multi-modal learing using Tacotron

An implementation of modified version of the Tacotron speech synthesis model in TensorFlow to generate bird-chirps (audio) samples given an image of a bird.



# Dataset
Folder structure

    
    ├── raw_data
          ├── imgs              # Folder contains all the images of the birds from 6 different breeds
          │   ├── 0             # Duck
          │   ├── 1             # Hawk
          │   ├── 2             # Owl
          │   ├── 3             # Seagull
          │   └── 4             # Macaw
          │   └── 5             # Rooster
          └── wavs              # Folder contains all the sounds of the birds from 6 different breeds  
          │   ├── 0             # Duck
          │   ├── 1             # Hawk
          │   ├── 2             # Owl
          │   ├── 3             # Seagull
          │   └── 4             # Macaw
          │   └── 5             # Rooster
          └── .....    
