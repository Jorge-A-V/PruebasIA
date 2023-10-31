import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)

#definitions
batch_size = 64
context_length =256
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2

with open('input.txt', 'r', encoding='utf-8') as file:
    text = file.read();

tokens = sorted(list(set(text)))
numero_tokens = len(tokens)
32
char_to_int = { ch:i for i,ch in enumerate(tokens) }
int_to_char = { i:ch for i,ch in enumerate(tokens) }
encode = lambda str: [char_to_int[car] for car in str]
decode = lambda lint: [int_to_char[car] for car in lint] 

data = torch.tensor(encode(text), dtype=torch.long)

train_percentage = 0.7
train_total = int(train_percentage * len(data))
train_set = data[:train_total]
test_set = data[train_total:]

def get_batch(split):
    data = train_set if split == "train" else test_set
    context_offset_batch = torch.randint(len(data) - context_length, (batch_size, ))
    x = torch.stack([data[i:i+context_length] for i in context_offset_batch])
    y = torch.stack([data[i+1:i+context_length+1] for i in context_offset_batch])
    x, y = x.to(device), y.to(device)
    return x,y

@torch.no_grad()
def estimate_loss(epoch, max_context):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        #length_penalty = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        length_penalty = (logits.size(-1) - max_context) * (1 / epoch)
        out[split] = losses.mean() + length_penalty
    model.train()
    return out;

class Head(nn.Module):
    #head of attentioni
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones((context_length, context_length))))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) # B T head_size
        q = self.query(x) # B T head_size

        wei = q @ k.transpose(-2,-1) * C**-0.5# B T C @ B C T --> B T T 

        #tril = torch.tril(torch.ones(T, T))
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList( [Head(head_size) for _ in range(num_heads)] )
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out =  torch.cat( [ h(x) for h in self.heads], dim=-1 )
        out = self.proj(out)
        out = self.dropout(out)
        return out

class FeedForward(nn.Module):
    #simple layer followd by a non linearily
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),#multiply by 4 the size of the innner layer of ffwd
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd), #proyection layer + cutting by 4 the size back
            nn.Dropout(dropout), #
        )
    
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa_heads = MultiHeadAttention(n_head, head_size)
        self.feed_forward = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd) #since n_embd = 32 --> both batch and time act as batch dimentions (B, T)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x): #forking and computing + layernorming beffore entering other blocks
        x = x + self.sa_heads(self.ln1(x)) #aplly self attention
        x = x+ self.feed_forward(self.ln2(x)) # B T C
        return x

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(numero_tokens, n_embd)
        self.position_embedding_table = nn.Embedding(context_length, n_embd)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head=n_head) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd) #final lYN
        self.lm_head = nn.Linear(n_embd, numero_tokens)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # batch by time by channel table (B, T, C)
        token_embedding = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # T C
        x = token_embedding + pos_emb # B T C
        x = self.blocks(x) #B T C
        #x = self.ln_f(x) ------------- * es lo mismo creo (cambiar si no furrula)
        logits = self.lm_head(self.ln_f(x)) # B T numero_tokens
        
        neuron_activations = []
        for block in self.blocks:
            block_layer = block(x)
            neuron_activations.append(block_layer.clone().detach())

        if targets is None:
            loss = None
        else:
            #Hay que transformarla para que funcione segun la documentacion a B*T, C
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T) #o (-1)
            loss = F.cross_entropy(logits, targets)
        return logits, loss, neuron_activations
    
    #coge B T --> lo tranorma a B T+1
    def generate(self, idx, max_new_tokens):
        #idx es el contexto B T
        for _ in range(max_new_tokens):
            #crop the context (recortar al contexto)
            idx_cond = idx[:, -context_length:]
            #prerdicciones
            logits, loss, neurons = self(idx_cond)
            #cogemos la ultima de todas
            #porque es lo que viene ahora
            logits = logits[:,-1,:] #B C
            #softmax a las probabilidades
            probs = F.softmax(logits, dim=-1) # B C
            #cogemos un sample de la distribucion --> SOLO 1 prediccion
            idx_next = torch.multinomial(probs, num_samples=1)# B 1
            #realizamos un append del sample a la secuencia B T+1
            idx=torch.cat((idx, idx_next), dim=1)
        return idx, neurons

tokens_in, targets = get_batch("train")
model = BigramLanguageModel()
m = model.to(device)
logits, loss = m(tokens_in, targets)



def iterate():
    for iter in range(max_iters):
        if iter % eval_interval == 0:
            losses = estimate_loss()
            print(
                f"step {iter} train loss {losses['train']:.4f} val loss {losses['val']:.4f}"
            )

iterate()

somecall, neurons = m.generate(
            torch.zeros((1,1), dtype=torch.long, device=device), max_new_tokens=100
            )[0].tolist()

print(
    decode(
        
    )
)

print(neurons[0])

optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)