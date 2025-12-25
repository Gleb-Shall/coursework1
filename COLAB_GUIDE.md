# Google Colab Setup Guide

## Quick Start

1. **Open Google Colab**: https://colab.research.google.com/

2. **Enable GPU**:
   - Runtime → Change runtime type → Hardware accelerator → GPU (T4)

3. **Upload the notebook**:
   - File → Upload notebook → Select `colab_setup.ipynb`

4. **Or create new notebook and copy cells from `colab_setup.ipynb`**

## Steps

### Option A: Upload Project Files

1. Zip the project folder:
   ```bash
   zip -r coursework1.zip coursework1/ -x "*.git*" "data/*" "models/*" "logs/*"
   ```

2. In Colab, upload the zip file:
   ```python
   from google.colab import files
   uploaded = files.upload()
   !unzip coursework1.zip
   %cd coursework1
   ```

### Option B: Clone from GitHub (if you have a repo)

```python
!git clone https://github.com/yourusername/coursework1.git
%cd coursework1
```

## Advantages of Colab

- ✅ **Free GPU** (T4, ~15GB VRAM)
- ✅ **Faster training** (10-20x faster than CPU)
- ✅ **No local setup needed**
- ✅ **Easy sharing**

## Notes

- Colab sessions timeout after ~12 hours of inactivity
- Free GPU has usage limits (check Colab quotas)
- Download models before session ends
- Data is lost when session ends (save to Drive or download)

## Save to Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')

# Save to Drive
!cp -r models/checkpoint/final_model /content/drive/MyDrive/
!cp -r results/ /content/drive/MyDrive/
```

## Troubleshooting

- **Out of memory**: Reduce `batch_size` to 4 or 2
- **Session timeout**: Save checkpoints frequently
- **Import errors**: Make sure all files are uploaded correctly

