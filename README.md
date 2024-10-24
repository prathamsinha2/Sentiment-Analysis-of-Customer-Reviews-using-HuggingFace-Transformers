# Review Sentiment Analyzer

A Python application that performs sentiment analysis on product reviews using the DistilBERT model. The tool analyzes text reviews from a CSV file and generates a visual representation of sentiment distribution through a pie chart.

## Features

- Sentiment analysis using DistilBERT (fine-tuned for sentiment classification)
- Processing of CSV files containing review data
- Visual representation of results using pie charts
- Progress tracking during analysis
- Handles invalid or empty reviews gracefully

## Prerequisites

```bash
pip install pandas matplotlib transformers torch
```

## Usage

1. Prepare your CSV file:
   - Must contain a column named `reviewText`
   - Each row should contain a text review

2. Run the application:
```bash
python app.py
```

3. When prompted, enter the path to your CSV file.

## How it Works

1. The application loads reviews from the specified CSV file
2. Each review is processed through the DistilBERT model for sentiment analysis
3. Results are classified as either POSITIVE or NEGATIVE with confidence scores
4. A pie chart is generated showing the distribution of sentiments

## Output

- Console output shows progress and individual sentiment results
- A pie chart visualization displays the overall sentiment distribution
- Invalid reviews are logged but skipped during processing

## Limitations

- Reviews are truncated to 512 tokens due to model constraints
- Empty or non-string reviews are automatically skipped
- Requires a valid CSV file with 'reviewText' column

## Technologies Used

- Python 3.x
- pandas: Data handling
- matplotlib: Visualization
- transformers: Hugging Face Transformers for sentiment analysis
- DistilBERT: Pre-trained model fine-tuned for sentiment classification

## Contributing

Feel free to open issues or submit pull requests with improvements.
