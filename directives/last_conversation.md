Ran command: `python -m src.data_gen.generate_dataset --count 2500 --balanced`

The script automatically splits the generated dataset into a training set and a validation set so that the YOLO model can test its performance on "unseen" images during training.

If you look closely at the console output you just pasted:
`Generating 2500 images (2125 train, 375 val)...`

The script has a default validation split of `15%`. 
- **2125 images** (85%) went into the `data/dataset/train/images` folder to be used for training.
- **375 images** (15%) went into the `data/dataset/val/images` folder to be used for validation.

All 2,500 images were generated successfully, they are just divided between those two folders! You are good to launch `python -m src.model.train`.