#test_abinet.py

import torch
from strhub.data.module import SceneTextDataModule
from strhub.data.dataset import LmdbDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import json

# Load pretrained ABINet from Torch Hub
abinet = torch.hub.load('baudm/parseq', 'abinet', pretrained=True).eval()
print("ABINet loaded successfully.")

# Path to LMDB test dataset
lmdb_path = '/home/d510/parseq_project/parseq/data/real/trainimg/lmdb'

# Load LMDB dataset
dataset = LmdbDataset(lmdb_path, transform=SceneTextDataModule.get_transform(abinet.hparams.img_size))
loader = DataLoader(dataset, batch_size=1, shuffle=False)

# Run inference
results = []
with torch.no_grad():
    for img, label_gt in tqdm(loader, desc="Testing ABINet"):
        logits = abinet(img)
        pred_probs = logits.softmax(-1)
        pred_label, confidence = abinet.tokenizer.decode(pred_probs)
        results.append({
            'gt': label_gt[0],
            'pred': pred_label[0],
            'confidence': confidence[0]
        })

# Optional: Print first 10 results
for r in results[:10]:
    print(f"GT: {r['gt']}, Pred: {r['pred']}, Conf: {r['confidence']:.2f}")

# Optional: Save results to JSON
with open('abinet_test_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("Testing completed. Results saved to abinet_test_results.json")
