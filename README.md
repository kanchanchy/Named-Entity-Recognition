# Named Entity Recognition and Classification

## Datasets
Two types of datasets are used in this entity recognition and classification task: CoNLL-2003 dataset and OntoNotes-5.0 dataset. For CoNLL_2003 dataset, raw dataset has been provided under data/conll_2003 folder. It contains all splits of the dataset for training, validation, and testing. For OntoNotes-5.0 dataset, since it is a licensed dataset, it is not provided here directly. You need to get the dataset following the guidelines from the website https://catalog.ldc.upenn.edu/LDC2013T19.

## LUKE Pretraining Model with Dice Loss
The goal here is to evaluate the LUKE pretraining model for NER task with the dice loss. In order to evaluate LUKE model with dice loss, I modified the luke model codes available in the repository https://github.com/studio-ousia/luke. The implementation of dice loss at the repository https://github.com/ShannonAI/dice_loss_for_NLP was utilized here.

### Data Preprocessing
Data preprocessing steps are performed automatically once the python file for training the code is run using command in the following step.

### Training
The following command should be used to train the model:

python -m Luke-Pretraining-Dice.ner_model.train_model --model-file=pretraining_model/luke_base_500k.tar.gz --output-dir=output ner run --data-dir=data/conll_2003 --train-batch-size=1 --gradient-accumulation-steps=2 --learning-rate=1e-5 --num-train-epochs=3 --fp16

You can change the parameters based on your need.

## LUKE-MRC with Dice Loss
The goal here is to evaluate BERT-MRC model for NER task with the entity representation model LUKE. Dice loss used by the original work is unchanged here. In order to evaluate LUKE model with dice loss, I modified the BERT-MRC model codes available in the repository https://github.com/ShannonAI/mrc-for-flat-nested-ner. The implementations of dice loss at the repository https://github.com/ShannonAI/dice_loss_for_NLP and LUKE pretraining model at the repository https://github.com/studio-ousia/luke were utilized here.

### Data Preprocessing
Data preprocessing steps from the repository https://github.com/ShannonAI/mrc-for-flat-nested-ner was utilized. Follow the steps in the repository to preprocess the dataset or download the preprocessed dataset from https://github.com/ShannonAI/mrc-for-flat-nested-ner/blob/master/ner2mrc/download.md.

### Training
Run Luke-MRC-Dice/scripts/luke_mrc.sh to train the model. You can change the parameters according to your needs on the .sh file.

### Evaluation
To evaluate a model after a checkpoint, run the following command:

CUDA_VISIBLE_DEVICES=0 python3 ${REPO_PATH}/tasks/mrc_ner/evaluate.py --gpus="1" --path_to_model_checkpoint $OUTPUT_DIR/epoch=2.ckpt

You need to make necessary changes based on the path on your system and availability of GPU.

