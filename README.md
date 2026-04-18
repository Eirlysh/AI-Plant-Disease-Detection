# AI-Plant-Disease-Detection
## Project Overview
This project focuses on classifying plant leaf diseases using deep learning techniques. The model is designed to achieve high accuracy while maintaining strong generalization.

## Methodology
- Pretrained EfficientNet-B0
- Two-stage training:
  - Stage 1: Train classifier (freeze backbone)
  - Stage 2: Fine-tune deeper layers
- Data augmentation + MixUp
- AdamW optimizer with cosine learning rate

## Results
- Test Accuracy: **99.30%**
- Top-3 Accuracy: **99.97%**
- Stable performance with minimal overfitting

### Model Confidence
The model demonstrates well-calibrated confidence:
- **Avg Confidence:** 0.9903 
- **Min Confidence:** 0.3824 
- **Max Confidence:** 1.0000  

## Limitations
- Dataset has controlled conditions (clean background)
- Real-world performance may slightly decrease

## Future Work
- Test on real-world images
- Improve robustness with stronger augmentation
- Streamlit deployment for plant disease classification  
