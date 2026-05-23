# SMS Spam Detection with PyTorch

A complete machine learning project for classifying SMS messages as spam or legitimate (ham) using a neural network built with PyTorch. This project demonstrates text preprocessing, data augmentation, and deep learning techniques for binary classification.

## Project Overview

This project implements a Multi-Layer Perceptron (MLP) neural network to classify SMS messages as either spam or legitimate messages. The model is trained on the UCI SMS Spam Collection dataset and achieves 99%+ accuracy after data augmentation.

## Features

- **Text Preprocessing**: Bag-of-Words vectorization with TF-IDF concepts
- **Data Augmentation**: Techniques to balance imbalanced datasets:
  - Random synonym insertion
  - Random word swapping
  - Random word deletion
- **Neural Network**: 3-layer MLP with batch normalization and dropout regularization
- **Comprehensive Evaluation**: Multiple metrics including accuracy, precision, recall, F1-score, MAE, and RMSE
- **Visualization**: Training curves, confusion matrix, and class distribution analysis

## Dataset

- **Source**: UCI SMS Spam Collection Dataset (https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- **Size**: 5,572 messages (original), 9,597 messages (after augmentation)
- **Class Distribution**:
  - Original: 87% Ham (legitimate), 13% Spam
  - After Augmentation: ~50% Ham, ~50% Spam (balanced)
- **Features**: Text messages up to 3000 words/bigrams

## Model Architecture

```
Input (3000) 
    ↓
FC1 (256) → BatchNorm → ReLU → Dropout(0.5)
    ↓
FC2 (128) → BatchNorm → ReLU → Dropout(0.3)
    ↓
FC3 (64) → BatchNorm → ReLU → Dropout(0.2)
    ↓
Output (2 classes: Ham/Spam)
```

**Model Statistics**:
- Total Parameters: ~1.1 million
- Trainable Parameters: ~1.1 million
- Optimizer: Adam (learning_rate=0.001)
- Loss Function: CrossEntropyLoss
- Epochs: 20
- Batch Size: 64

## Results

### Before Data Augmentation
- Train Accuracy: 99%+
- Test Accuracy: 98%
- Signs of overfitting: Large gap between train and test loss

### After Data Augmentation
- Train Accuracy: 99-99.5%
- Test Accuracy: 99-99.3%
- **Eliminated overfitting**: Train and test curves closely aligned
- **Improved stability**: Consistent performance across epochs

### Final Metrics (After Augmentation)
| Metric | Score |
|--------|-------|
| Accuracy | 0.9937 |
| Precision | 0.9884 |
| Recall | 0.9831 |
| F1 Score | 0.9857 |
| MAE | 0.0063 |
| RMSE | 0.0793 |

## Installation

### Requirements
- Python 3.7+
- PyTorch 2.0+
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- nltk

### Setup

```bash
# Clone the repository
git clone https://github.com/KhadijaKammounIDSD/SMS-Spam-Detection.git
cd SMS-Spam-Detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch torchvision torchaudio
pip install scikit-learn pandas numpy matplotlib seaborn nltk

# Download NLTK data (required for augmentation)
python -c "import nltk; nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger'); nltk.download('omw-1.4')"
```

## Usage

### Training the Model

Open and run the Jupyter notebook with all training steps:

```bash
jupyter notebook SMS_Spam_Detection_After_DataAugmentation.ipynb
```

The notebook includes:
- Data loading and preprocessing
- Data augmentation
- Model training
- Evaluation and visualization

### Making Predictions

```python
def predict_message(message, model, vectorizer):
    """Predict if a message is spam or ham"""
    message_vec = vectorizer.transform([message]).toarray()
    message_tensor = torch.FloatTensor(message_vec)
    
    model.eval()
    with torch.no_grad():
        output = model(message_tensor)
        probabilities = torch.softmax(output, dim=1)
        _, predicted = torch.max(output, 1)
    
    label = "SPAM" if predicted.item() == 1 else "HAM"
    confidence = probabilities[0][predicted.item()].item() * 100
    
    return label, confidence

# Example usage
message = "Congratulations! You've won a free iPhone!"
label, confidence = predict_message(message, model, vectorizer)
print(f"Prediction: {label} (Confidence: {confidence:.2f}%)")
```

## Project Structure

```
sms-spam-detection/
├── README.md                                              # Project documentation
├── SMS_Spam_Detection_After_DataAugmentation.ipynb        # Main notebook (with augmentation)
├── sms_spam_detection_beforeDataAugmentation.ipynb        # Reference notebook (before augmentation)
├── spam.csv                                               # Original dataset (5,572 messages)
└── spam_classifier.pth                                    # Trained model weights
```

## Key Insights

1. **Data Imbalance Problem**: Original dataset had 87% ham vs 13% spam, which can bias the model
2. **Data Augmentation Solution**: Synthetic augmentation of minority class creates balanced training data
3. **Overfitting Reduction**: Data augmentation significantly reduced the gap between training and test performance
4. **Regularization Techniques**: Dropout and batch normalization further improve generalization
5. **Model Performance**: 99%+ accuracy achieved on test set with minimal overfitting

## Data Augmentation Techniques

The project implements three augmentation strategies:

1. **Random Insertion**: Insert synonyms of random words
   - Example: "Call now" → "Call now telephone"

2. **Random Swap**: Randomly swap positions of words
   - Example: "Free money" → "Money free"

3. **Random Deletion**: Randomly remove words with probability p
   - Example: "Click here now" → "Click here" (now deleted)

These techniques preserve semantic meaning while creating data diversity.

## Comparison: Before vs After Augmentation

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Dataset Size | 5,572 | 9,597 | +72% |
| Train Loss | ~0.001 | ~0.015-0.020 | More stable |
| Test Loss | ~0.09 | ~0.015-0.025 | 82% reduction |
| Overfitting Gap | 0.089 | ~0.005 | 94% reduction |
| Class Balance | 87-13 | 50-50 | Perfectly balanced |

## Future Improvements

- [ ] Implement transformer-based models (BERT, DistilBERT)
- [ ] Add real-time prediction API using Flask/FastAPI
- [ ] Deploy model using Docker containers
- [ ] Implement cross-validation for more robust evaluation
- [ ] Add support for multi-language spam detection
- [ ] Create web interface for predictions
- [ ] Implement class weighting for imbalanced data

## Troubleshooting

**Issue**: NLTK data not found
```python
import nltk
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')
```

**Issue**: Out of memory during augmentation
- Reduce `max_features` in CountVectorizer
- Use smaller batch sizes
- Process data in chunks

## References

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [UCI SMS Spam Collection Dataset](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection)
- [Data Augmentation for Text Classification](https://arxiv.org/abs/1901.11196)

## Author

**Khadija Kammoun** - Deep Learning Course Project 

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- UCI Machine Learning Repository for the dataset
- PyTorch team for the excellent deep learning framework
- scikit-learn for text preprocessing and evaluation metrics
- Course instructors for project guidance

---

## Contact

For questions, issues, or suggestions:
- GitHub Profile: [KhadijaKammounIDSD](https://github.com/KhadijaKammounIDSD)
- GitHub Issues: [Create an issue](https://github.com/KhadijaKammounIDSD/SMS-Spam-Detection/issues)

**Last Updated**: May 23, 2026#
