from pathlib import Path
import json
from sentence_transformers import (
    InputExample,
    datasets,
    SentenceTransformer,
    losses
)

# get all jsonl filepaths
paths = [str(path) for path in Path('./data').glob('*.jsonl')]

lines = []
# extract all jsonl into same lines list
for path in paths:
    with open(path, 'r') as fp:
        lines.extend([json.loads(record) for record in list(fp)])

# create input example from question, context pairs
train = []
for line in lines:
    train.append(InputExample(
        texts=[line['question'], line['context']]
    ))

# using MNR loss we should ensure each batch does not include duplicates
batch_size = 24
loader = datasets.NoDuplicatesDataLoader(
    train, batch_size=batch_size
)

# init an existing sentence transformer (as we have little training data)
model = SentenceTransformer('pinecone/bert-retriever-squad2')

# init MNR loss for training
loss = losses.MultipleNegativesRankingLoss(model)

# and train
epochs = 1
warmup_steps = int(len(loader) * epochs * 0.1)
model.fit(
    train_objectives=[(loader, loss)],
    epochs=epochs,
    warmup_steps=warmup_steps,
    output_path='./models/mpnet-mnr-discourse',
    show_progress_bar=True
)