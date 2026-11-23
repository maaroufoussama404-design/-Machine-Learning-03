import os, time, json, statistics
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torchinfo import summary
from thop import profile

ROOT = Path(__file__).resolve().parent
for d in ["data","src","results","reports"]:
    (ROOT/d).mkdir(exist_ok=True)

torch.manual_seed(42)
np.random.seed(42)
torch.set_num_threads(min(8, os.cpu_count() or 1))
DEVICE = "cpu"
BATCH, EPOCHS = 128, 5

print("Loading FashionMNIST...")
tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,))
])
full = datasets.FashionMNIST(ROOT/"data", train=True, download=True, transform=tf)
test = datasets.FashionMNIST(ROOT/"data", train=False, download=True, transform=tf)
train, val = random_split(full, [55000,5000], generator=torch.Generator().manual_seed(42))
train_loader = DataLoader(train, batch_size=BATCH, shuffle=True, num_workers=0)
val_loader = DataLoader(val, batch_size=BATCH, shuffle=False, num_workers=0)
test_loader = DataLoader(test, batch_size=BATCH, shuffle=False, num_workers=0)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(),nn.Linear(784,256),nn.ReLU(),
                                 nn.Linear(256,128),nn.ReLU(),nn.Linear(128,10))
    def forward(self,x): return self.net(x)

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1,16,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2))
        self.fc = nn.Linear(32*7*7,10)
    def forward(self,x): return self.fc(self.features(x).flatten(1))

class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(49,64)
        layer = nn.TransformerEncoderLayer(64,4,128,batch_first=True)
        self.encoder = nn.TransformerEncoder(layer,2)
        self.fc = nn.Linear(64,10)
    def forward(self,x):
        p = x.unfold(2,7,7).unfold(3,7,7)
        p = p.permute(0,2,3,1,4,5).contiguous().reshape(x.size(0),16,49)
        return self.fc(self.encoder(self.proj(p)).mean(1))

def evaluate(model, loader, criterion):
    model.eval()
    loss_sum, correct, total = 0,0,0
    with torch.no_grad():
        for x,y in loader:
            out=model(x)
            loss_sum += criterion(out,y).item()*y.size(0)
            correct += (out.argmax(1)==y).sum().item()
            total += y.size(0)
    return loss_sum/total, correct/total

def hand_flops(name):
    if name=="MLP":
        return 2*(784*256+256*128+128*10)
    if name=="CNN":
        return 2*(9*1*16*28*28+9*16*32*14*14+32*7*7*10)
    # Patch projection + 2 Transformer layers + classifier.
    s,d,f=16,64,128
    projection=2*s*49*d
    one_layer=2*s*d*3*d+2*s*d*d+4*s*s*d+4*s*d*f
    return projection+2*one_layer+2*d*10

histories=[]
results=[]
for name, Model in [("MLP",MLP),("CNN",CNN),("Attention",Attention)]:
    print(f"\nTraining {name}...")
    model=Model()
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)
    criterion=nn.CrossEntropyLoss()
    best=0
    checkpoint=ROOT/"results"/f"{name.lower()}_best.pth"
    for epoch in range(1,EPOCHS+1):
        model.train()
        start=time.perf_counter()
        for x,y in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss=criterion(model(x),y)
            loss.backward()
            optimizer.step()
        _,train_acc=evaluate(model,train_loader,criterion)
        val_loss,val_acc=evaluate(model,val_loader,criterion)
        histories.append({"model":name,"epoch":epoch,"train_accuracy":train_acc,
                          "validation_accuracy":val_acc,"validation_loss":val_loss})
        print(f"Epoch {epoch}/{EPOCHS}: train={train_acc:.2%}, val={val_acc:.2%}, {time.perf_counter()-start:.1f}s")
        if val_acc>best:
            best=val_acc
            torch.save(model.state_dict(),checkpoint)
    model.load_state_dict(torch.load(checkpoint,map_location="cpu",weights_only=True))
    test_loss,test_acc=evaluate(model,test_loader,criterion)

    params=sum(p.numel() for p in model.parameters())
    model.eval()
    try:
        macs,_=profile(model,inputs=(torch.randn(1,1,28,28),),verbose=False)
    except Exception as e:
        print("THOP warning:",e)
        macs=float("nan")

    latency={}
    with torch.no_grad():
        for bs,reps in [(1,100),(128,20)]:
            x=torch.randn(bs,1,28,28)
            for _ in range(10): model(x)
            samples=[]
            for _ in range(reps):
                t=time.perf_counter()
                model(x)
                samples.append((time.perf_counter()-t)*1000)
            latency[f"batch_{bs}_ms"]=statistics.median(samples)

    results.append({"model":name,"parameters":params,"hand_flops":hand_flops(name),
                    "thop_macs":float(macs),"test_accuracy":test_acc,
                    "test_loss":test_loss,**latency})
    (ROOT/"reports"/f"{name.lower()}_summary.txt").write_text(
        str(summary(model,input_size=(1,1,28,28),verbose=0)),encoding="utf-8")
    print(f"{name}: params={params:,}, test accuracy={test_acc:.2%}")

df=pd.DataFrame(results)
hist=pd.DataFrame(histories)
df.to_csv(ROOT/"results"/"model_comparison.csv",index=False)
hist.to_csv(ROOT/"results"/"training_history.csv",index=False)
(ROOT/"results"/"model_comparison.json").write_text(df.to_json(orient="records",indent=2))

for x,y,title,filename in [
    ("parameters","test_accuracy","Accuracy vs parameters","accuracy_vs_parameters.png"),
    ("batch_1_ms","test_accuracy","Accuracy vs latency (batch 1)","accuracy_vs_latency.png")]:
    plt.figure(figsize=(8,5))
    for _,r in df.iterrows():
        plt.scatter(r[x],r[y]*100,label=r["model"])
        plt.annotate(r["model"],(r[x],r[y]*100))
    plt.xlabel(x); plt.ylabel("Test accuracy (%)"); plt.title(title)
    plt.grid(alpha=.3); plt.legend(); plt.tight_layout()
    plt.savefig(ROOT/"results"/filename,dpi=160); plt.close()

plt.figure(figsize=(8,5))
for name in hist.model.unique():
    h=hist[hist.model==name]
    plt.plot(h.epoch,h.validation_accuracy*100,marker="o",label=name)
plt.xlabel("Epoch"); plt.ylabel("Validation accuracy (%)")
plt.title("Validation accuracy by epoch"); plt.legend(); plt.grid(alpha=.3)
plt.tight_layout(); plt.savefig(ROOT/"results"/"validation_accuracy.png",dpi=160); plt.close()

table=df.to_markdown(index=False,floatfmt=".4f")
best=df.loc[df.test_accuracy.idxmax()]
fast=df.loc[df.batch_1_ms.idxmin()]
small=df.loc[df.parameters.idxmin()]
report=f"""# Lab 3: Architectures and their costs

## 1. Objective
Compare MLP, CNN and patch-based Transformer architectures on FashionMNIST,
considering accuracy, trainable parameters, FLOPs and inference latency.

## 2. Dataset and setup
- FashionMNIST: 55,000 training, 5,000 validation, 10,000 official test images.
- Images: 28x28 grayscale, ToTensor and Normalize((0.2860,), (0.3530,)).
- Random seed: 42. Batch size: 128. Epochs: 5.
- Optimizer: Adam, learning rate 0.001. Loss: cross entropy.
- Runtime: CPU, PyTorch {torch.__version__}.

## 3. Architectures
- **MLP:** Flatten 784 -> Linear 256 -> ReLU -> Linear 128 -> ReLU -> Linear 10.
- **CNN:** Conv(1,16,3,padding=1) -> ReLU -> MaxPool(2) ->
  Conv(16,32,3,padding=1) -> ReLU -> MaxPool(2) -> Linear(1568,10).
- **Attention:** 16 non-overlapping 7x7 patches; projection 49->64;
  two Transformer encoder layers (64 dimensions, 4 heads, feed-forward 128);
  mean pooling and Linear(64,10).

## 4. Results
{table}

## 5. FLOP methodology
One multiply-accumulate is counted as two FLOPs. The hand calculations include
dense and convolution operations, and for attention include patch projection,
Q/K/V projections, output projection, attention score/value products,
feed-forward layers and classifier. Biases, normalization, activations,
softmax, residuals and pooling are omitted. THOP reports MACs for operations
it recognizes; multiplying MACs by two gives an approximate FLOP count.
Attention internals may not be fully represented by THOP, so the two estimates
can differ.

## 6. Architectural and system trade-offs
- **MLP:** dense layers mix all input positions, but flattening removes explicit
  spatial structure. Dense operations have regular matrix access patterns.
- **CNN:** local filters share weights across image positions and exploit
  spatial locality. Pooling reduces feature-map size and later computation.
  Convolution performance benefits from data reuse and efficient memory layouts.
- **Attention:** patches become a token sequence. Each token can interact with
  all other tokens, capturing global relationships. Attention score computation
  grows quadratically with sequence length.
- **System level:** parameter count and FLOPs do not fully determine latency.
  Memory traffic, cache behavior, tensor layout, batch size and kernel
  implementation also affect runtime. Batch 1 measures single-input latency;
  batch 128 gives a different throughput-oriented workload.

## 7. Interpretation
Best measured test accuracy: **{best.model} ({best.test_accuracy:.2%})**.
Fastest batch-1 inference: **{fast.model} ({fast.batch_1_ms:.3f} ms median)**.
Smallest parameter count: **{small.model} ({int(small.parameters):,})**.

The assignment's suggested 100k-150k parameter range conflicts with the
explicit architectures: the specified MLP alone has more parameters than
that range. This implementation follows the explicit layer definitions
rather than changing the requested models.

## 8. Output files
Comparison CSV/JSON, training history, plots and model checkpoints are in
`results/`; model summaries are in `reports/`; dataset files are in `data/`.
Latency is measured as median wall-clock forward-pass time after 10 warm-ups:
100 measurements for batch 1 and 20 for batch 128. Results are specific to
this CPU and software environment.
"""
(ROOT/"report.md").write_text(report,encoding="utf-8")
(ROOT/"reports"/"report.md").write_text(report,encoding="utf-8")
print("\nLAB 3 COMPLETE")
print("Report:",ROOT/"report.md")
print("Results:",ROOT/"results")
print(df.to_string(index=False))
