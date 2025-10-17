# Handling Long Sequences in User Identification

## Problem Overview

**Current situation:**
- Average sequence length: **~800 actions**
- Current model: Attention-LSTM with max_sequence_length=1000
- Challenge: Standard LSTMs struggle with sequences this long

**Why this matters:**
- Long sequences → vanishing/exploding gradients
- Attention becomes diluted over 800+ timesteps
- Higher memory usage and slower training
- Potential information loss with truncation

---

## Solution Options

### **Option A: Smart Truncation Strategy** ⚡ (Quick Fix - 5 minutes)

**Implementation:** Adjust which part of sequence to keep

**Strategy 1: Keep Most Recent Actions**
```yaml
# config/config_attention.yaml
data:
  max_sequence_length: 600  # Keep last 600 actions
```

**Rationale:** Recent behavior is often most indicative of user identity

**Strategy 2: First + Last Actions**
```python
# In utils.py - prepare_rnn_sequences()
# Keep first 100 + last 500 actions
if len(id_sequence) > 600:
    first_part = id_sequence[:100]
    last_part = id_sequence[-500:]
    id_sequence = first_part + last_part
```

**Rationale:** Captures session start pattern + recent behavior

**Strategy 3: Sampling**
```python
# Sample every Nth action for very long sequences
if len(id_sequence) > 1000:
    step = len(id_sequence) // 600
    id_sequence = id_sequence[::step]
```

**Rationale:** Maintains temporal coverage while reducing length

**Pros:**
- ✅ Immediate implementation (just change config)
- ✅ No architecture changes needed
- ✅ Lower memory usage
- ✅ Faster training

**Cons:**
- ❌ Potential information loss
- ❌ May miss important mid-sequence patterns

**Expected Performance:** 50-65% accuracy (baseline for comparison)

---

### **Option B: Hierarchical/Chunked Processing** 📊 (Moderate - 30 minutes)

**Implementation:** Process sequence in chunks, then aggregate

**Architecture:**
```python
# Split 800-action sequence into 8 chunks of 100 actions each
chunks = split_sequence(sequence, num_chunks=8, chunk_size=100)

# Process each chunk with BiLSTM
chunk_representations = []
for chunk in chunks:
    chunk_repr = BiLSTM(chunk)  # → (batch, hidden_size)
    chunk_representations.append(chunk_repr)

# Stack chunks: (batch, 8, hidden_size)
chunk_stack = torch.stack(chunk_representations, dim=1)

# Apply attention across chunks
context = Attention(chunk_stack)  # → (batch, hidden_size)

# Continue with classification
output = Classifier(context + stats_features + browser)
```

**Modifications needed:**
- Update `rnn_attention_lightning.py` to process chunks
- Add chunk splitting in forward pass
- Two-level attention: within chunks + across chunks

**Pros:**
- ✅ No information loss
- ✅ Scales well to any sequence length
- ✅ Better interpretability (chunk-level attention)
- ✅ More efficient than processing full sequence

**Cons:**
- ❌ Moderate code changes required
- ❌ Slightly more complex training

**Expected Performance:** 55-70% accuracy

---

### **Option C: Temporal Convolutional Network (TCN)** 🚀 (Best for Long Sequences - 1-2 hours)

**Implementation:** Replace LSTM with dilated convolutions

**Why TCN for long sequences:**
- Dilated convolutions have exponentially growing receptive field
- Parallelizable (unlike LSTM)
- More efficient memory usage
- Specifically designed for long sequences

**Architecture:**
```python
class TCNUserClassifier(pl.LightningModule):
    def __init__(self):
        self.embedding = nn.Embedding(vocab_size, 128)
        
        # TCN layers with increasing dilation
        self.tcn = nn.Sequential(
            TCNBlock(128, 256, kernel_size=3, dilation=1),   # Receptive field: 3
            TCNBlock(256, 256, kernel_size=3, dilation=2),   # Receptive field: 9
            TCNBlock(256, 256, kernel_size=3, dilation=4),   # Receptive field: 27
            TCNBlock(256, 256, kernel_size=3, dilation=8),   # Receptive field: 81
            TCNBlock(256, 256, kernel_size=3, dilation=16),  # Receptive field: 243
            TCNBlock(256, 256, kernel_size=3, dilation=32),  # Receptive field: 729
        )
        
        self.attention = Attention(256, 128)
        self.classifier = nn.Linear(256 + stat_features + browser, num_users)
    
    def forward(self, x):
        embedded = self.embedding(x)
        tcn_out = self.tcn(embedded)
        context = self.attention(tcn_out)
        return self.classifier(context + features)
```

**TCN Block:**
```python
class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               padding=(kernel_size-1)*dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=(kernel_size-1)*dilation, dilation=dilation)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.dropout(out)
        out = self.relu(self.conv2(out))
        out = self.dropout(out)
        return out + x  # Residual connection
```

**Files to create:**
- `tcn_model.py` - TCN architecture
- `train_tcn_lightning.py` - Training script
- `config/config_tcn.yaml` - TCN-specific config

**Pros:**
- ✅ Excellent for long sequences (designed for this)
- ✅ Faster training than LSTM
- ✅ Better gradient flow
- ✅ Parallelizable
- ✅ Lower memory usage than LSTM

**Cons:**
- ❌ Requires new architecture implementation
- ❌ Different from familiar LSTM paradigm

**Expected Performance:** 60-75% accuracy (+5-10% over LSTM)

---

### **Option D: CNN-BiLSTM-Attention Hybrid** 🎯 (Best Balance - 1 hour)

**Implementation:** Use CNN to compress sequence, then LSTM+Attention

**Architecture:**
```python
class CNNLSTMAttentionClassifier(pl.LightningModule):
    def __init__(self):
        self.embedding = nn.Embedding(vocab_size, 128)
        
        # 1D CNN to extract local patterns and reduce sequence length
        self.cnn = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2),  # 800 → 400
            nn.ReLU(),
            nn.MaxPool1d(2),  # 400 → 200
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        # Output: sequence length reduced from 800 → 200
        
        # BiLSTM on compressed representation
        self.lstm = nn.LSTM(256, 256, num_layers=2, bidirectional=True, batch_first=True)
        
        # Attention over LSTM outputs
        self.attention = BahdanauAttention(512, 128)  # 512 = 256*2 (bidirectional)
        
        # Fusion and classification
        self.fusion = nn.Sequential(
            nn.Linear(512 + stat_features + browser, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
        )
        self.classifier = nn.Linear(256, num_users)
    
    def forward(self, sequences, stat_features, browser_features):
        # Embed: (batch, 800) → (batch, 800, 128)
        embedded = self.embedding(sequences)
        
        # CNN expects (batch, channels, length)
        embedded = embedded.permute(0, 2, 1)  # → (batch, 128, 800)
        
        # CNN compression: (batch, 128, 800) → (batch, 256, 200)
        cnn_out = self.cnn(embedded)
        
        # Back to (batch, length, channels) for LSTM
        cnn_out = cnn_out.permute(0, 2, 1)  # → (batch, 200, 256)
        
        # BiLSTM: (batch, 200, 256) → (batch, 200, 512)
        lstm_out, _ = self.lstm(cnn_out)
        
        # Attention: (batch, 200, 512) → (batch, 512)
        context, attention_weights = self.attention(lstm_out)
        
        # Fusion and classification
        combined = torch.cat([context, stat_features, browser_features], dim=1)
        fused = self.fusion(combined)
        logits = self.classifier(fused)
        
        return logits, attention_weights
```

**Key Benefits:**
- **CNN layer**: Reduces 800 → 200 sequence length (4x compression)
- **Local patterns**: CNN captures n-gram-like action patterns
- **BiLSTM**: Models temporal dependencies on compressed sequence
- **Attention**: Focuses on important compressed segments
- **Best of both worlds**: Speed of CNN + modeling power of LSTM

**Pros:**
- ✅ Significant speedup (4x fewer LSTM steps)
- ✅ Better feature extraction (CNN finds local patterns)
- ✅ Still uses proven LSTM+Attention
- ✅ Moderate implementation effort
- ✅ Lower memory usage

**Cons:**
- ❌ More hyperparameters to tune
- ❌ Need to balance CNN vs LSTM capacity

**Expected Performance:** 60-75% accuracy

---

### **Option E: Transformer with Sparse Attention** 🔬 (Advanced - 2-3 hours)

**Implementation:** Transformer with efficient attention for long sequences

**Architecture:**
```python
class TransformerUserClassifier(pl.LightningModule):
    def __init__(self):
        self.embedding = nn.Embedding(vocab_size, 128)
        self.pos_encoding = PositionalEncoding(128, max_len=1000)
        
        # Transformer with local + global attention
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=8,
            dim_feedforward=512,
            dropout=0.3,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        # Global pooling + attention
        self.attention = nn.MultiheadAttention(128, 8, batch_first=True)
        
        self.classifier = nn.Linear(128 + stat_features + browser, num_users)
```

**Sparse Attention Options:**
- Local attention windows (only attend to nearby actions)
- Strided attention (attend every Kth position)
- Random attention (mix of local + random global)

**Pros:**
- ✅ State-of-the-art for sequences
- ✅ Fully parallelizable
- ✅ Better long-range modeling than LSTM
- ✅ Attention maps are interpretable

**Cons:**
- ❌ Complex implementation
- ❌ Quadratic memory in sequence length (needs sparse attention)
- ❌ Many hyperparameters
- ❌ Slower training without GPU

**Expected Performance:** 60-75% accuracy

---

### **Option F: Sequence Compression via N-grams** 🔤 (Preprocessing - 30 minutes)

**Implementation:** Reduce sequence length by grouping actions

**Strategy 1: Action Bigrams**
```python
# Instead of: [Action1, Action2, Action3, Action4, ...]
# Use bigrams: [Action1+Action2, Action2+Action3, Action3+Action4, ...]
# Sequence length reduces by ~50%

def create_ngram_vocabulary(sequences, n=2):
    """Create vocabulary of n-grams instead of single actions"""
    ngrams = set()
    for seq in sequences:
        for i in range(len(seq) - n + 1):
            ngram = tuple(seq[i:i+n])
            ngrams.add(ngram)
    return {ngram: idx for idx, ngram in enumerate(ngrams)}
```

**Strategy 2: Action Compression**
```python
# Merge consecutive duplicate or similar actions
# [Button, Button, Button, Dialog] → [Button_x3, Dialog]
# Can reduce sequence by 30-50%

def compress_sequence(sequence):
    """Merge consecutive similar actions"""
    compressed = []
    current_action = None
    count = 0
    
    for action in sequence:
        if action == current_action:
            count += 1
        else:
            if current_action:
                compressed.append((current_action, count))
            current_action = action
            count = 1
    
    if current_action:
        compressed.append((current_action, count))
    
    return compressed
```

**Pros:**
- ✅ Reduces sequence length significantly
- ✅ Captures action patterns (bigrams are meaningful)
- ✅ Works with any model architecture
- ✅ No model changes needed

**Cons:**
- ❌ Larger vocabulary size
- ❌ May lose some temporal information
- ❌ Requires vocabulary retraining

**Expected Performance:** 55-70% accuracy (similar to current)

---

## Comparison Table

| Option | Effort | Training Time | Memory | Accuracy | Best For |
|--------|--------|---------------|--------|----------|----------|
| **A: Smart Truncation** | 5 min | 20-30 min | Low | 50-65% | Quick test |
| **B: Hierarchical** | 30 min | 30-45 min | Medium | 55-70% | No info loss |
| **C: TCN** | 1-2 hr | 20-35 min | Low | 60-75% | Long sequences |
| **D: CNN-LSTM (Hybrid)** | 1 hr | 25-40 min | Medium | 60-75% | **Best balance** |
| **E: Transformer** | 2-3 hr | 40-60 min | High | 60-75% | SOTA research |
| **F: N-gram Compression** | 30 min | 30-45 min | Medium | 55-70% | Preprocessing |

---

## Recommendations

### **For Immediate Testing:**
1. **Start with Option A** - Change `max_sequence_length: 600` (keep most recent)
2. Run training and check if performance is acceptable
3. This establishes baseline with reduced sequence length

### **For Best Performance:**
**Recommended: Option D (CNN-LSTM Hybrid)**
- Best balance of implementation effort vs performance gain
- Uses proven architectures (CNN + LSTM + Attention)
- 4x speedup from CNN compression
- Expected 60-75% accuracy

### **If You Want Cutting Edge:**
**Option C (TCN)** - Purpose-built for long sequences
- More efficient than LSTM
- Better gradient flow
- Expected 60-75% accuracy
- Worth learning for future projects

### **If Resources are Limited:**
**Option A + F Combined**
- Keep most recent 600 actions (Option A)
- Use action bigrams (Option F)
- Minimal changes, decent performance

---

## Implementation Priority

### Phase 1: Quick Validation (Do First)
```yaml
# config/config_attention.yaml
data:
  max_sequence_length: 600  # Reduced from 1000
```
**Action:** Test current model with shorter sequences
**Time:** 5 minutes + 20-30 min training
**Goal:** Establish baseline

### Phase 2: If Phase 1 Works Well (50-60% accuracy)
**Stop here!** Current architecture is sufficient.

### Phase 3: If Phase 1 Needs Improvement
**Implement Option D (CNN-LSTM Hybrid)**
**Time:** 1 hour implementation + 30 min training
**Expected gain:** +10-15% accuracy

### Phase 4: If Maximum Performance Needed
**Implement Option C (TCN)** or ensemble multiple models
**Time:** 2-3 hours
**Expected:** 65-75% accuracy

---

## Code Examples Location

### Smart Truncation (Option A)
- **File:** `config/config_attention.yaml`
- **Change:** `max_sequence_length: 600`

### Hierarchical Processing (Option B)
- **File:** Create `rnn_hierarchical_lightning.py`
- **Based on:** `rnn_attention_lightning.py`

### TCN (Option C)
- **File:** Create `tcn_model.py` and `train_tcn_lightning.py`
- **Reference:** See architecture code above

### CNN-LSTM Hybrid (Option D)
- **File:** Create `cnn_lstm_attention_lightning.py`
- **Based on:** `rnn_attention_lightning.py` with CNN preprocessing

---

## Next Steps

1. **Test with reduced sequence length** (Option A)
2. **Analyze results**: 
   - If 55%+ accuracy → Current approach works
   - If <55% accuracy → Consider Option C or D
3. **Implement best option** based on results
4. **Tune hyperparameters**
5. **Evaluate final performance**

---

## Questions?

**Q: Which option should I use?**
A: Start with Option A (truncation). If you need better performance, go with Option D (CNN-LSTM).

**Q: Will I lose accuracy by truncating?**
A: Possibly 5-10%, but recent actions are often most indicative. Test to confirm.

**Q: Can I combine options?**
A: Yes! Option A (truncation) + Option F (n-grams) + Option B (hierarchical) work well together.

**Q: What about training time?**
A: Shorter sequences = faster training. 600 vs 1000 length saves ~30% time.

---

**Status:** Options documented. Ready to implement based on your choice.

