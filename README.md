## Basic Training Instructions
```bash
pip install -r requirements
python main.py --phase train --exp "test_exp" --nb_epochs 150
```

This should create a chekpoint directory under `ckdir/<modelID>/` where <modelID> is the ID
of the model created during training. 


## Basic Testing Instructions
```bash
python main.py --phase test --model_id <modelID>
```

This should evaluate the generated model against the test dataset.


## Basic Exporting Instructions
```bash
python export.py --model_id <modelID> --target saved_model
```

This exports the trained model for inference using saved_model method of tensorflow. It also tests the
model against the test dataset. The exported model ready to be used with tf-serving is located at 
`export_dir/<modelID>/serving`


## Help 
```bash
python main.py --h
```


