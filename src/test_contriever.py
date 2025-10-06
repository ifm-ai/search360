import torch
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

class SentenceDataset(Dataset):
    def __init__(self, sentences, tokenizer, max_length=512):
        self.sentences = sentences
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.sentences[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {k: v.squeeze(0) for k, v in encoding.items()}


# Initialize
print(f"Available GPUs: {torch.cuda.device_count()}")

tokenizer = AutoTokenizer.from_pretrained("facebook/contriever")
model = AutoModel.from_pretrained("facebook/contriever")

# Use DataParallel for 8 GPUs
if torch.cuda.device_count() >= 8:
    model = torch.nn.DataParallel(model, device_ids=list(range(8)))
elif torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)

model = model.to("cuda")
model.eval()

sentences = [
    "Where was Marie Curie born?",
    "Maria Sklodowska, later known as Marie Curie, was born on November 7, 1867.",
    "Born in Paris on 15 May 1859, Pierre Curie was the son of Eugène Curie, a doctor of French Catholic origin from Alsace.",
] * 10000  # Scale up for better GPU utilization

# Create dataset and dataloader
dataset = SentenceDataset(sentences, tokenizer)
dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)


# Mean pooling function
def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.0)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings


# Process batches
all_embeddings = []
with torch.no_grad():
    for batch in tqdm(dataloader):
        inputs = {k: v.to("cuda") for k, v in batch.items()}
        outputs = model(**inputs)
        embeddings = mean_pooling(outputs[0], inputs["attention_mask"])
        all_embeddings.append(embeddings.cpu())

# Concatenate all embeddings
final_embeddings = torch.cat(all_embeddings, dim=0)
print(f"Final embeddings shape: {final_embeddings.shape}")
