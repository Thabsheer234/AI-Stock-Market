import torch

with open(r"C:\Stock_market_agentic_ai\main.py", "r", encoding="utf-8") as f:
    text = f.read()

# Add torch import if not present
if "import torch" not in text:
    text = "import torch\n" + text

# Replace hardcoded cuda
old_cuda = 'model_kwargs={"device": "cuda"}'
new_cuda = 'model_kwargs={"device": ("cuda" if torch.cuda.is_available() else "cpu")}'
if old_cuda in text:
    text = text.replace(old_cuda, new_cuda)
    print("Replaced model_kwargs device with CUDA/CPU fallback!")

# Remove hardcoded 2026-06-24 date in yfinance
if ', end="2026-06-24"' in text:
    text = text.replace(', end="2026-06-24"', '')
    print("Removed hardcoded end date in yfinance call!")

with open(r"C:\Stock_market_agentic_ai\main.py", "w", encoding="utf-8") as f:
    f.write(text)

print("main.py patched successfully!")
