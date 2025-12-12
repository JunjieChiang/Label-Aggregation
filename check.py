import numpy as np, os
root='metrics/trec2011'

logits=np.load(os.path.join(root,'object_logits.npy'))
gt_idx=np.load(os.path.join(root,'ground_truth_indices.npy'))
labels=np.load(os.path.join(root,'labels.npy'))
probs=np.exp(logits-logits.max(1,keepdims=True))
probs=probs/probs.sum(1,keepdims=True)
preds=logits.argmax(1)
rows=[(obj,true,preds[obj],probs[obj,true],probs[obj,preds[obj]]) for obj,true in zip(gt_idx,labels) if preds[obj]!=true]
print('wrong cases',len(rows))

for row in rows[:20]:
    print(row)