from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from transformers import pipeline
import pandas as pd
import matplotlib.pyplot as plt
import io
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

model = "distilbert-base-uncased-finetuned-sst-2-english"
sent_pl = pipeline("sentiment-analysis", model=model)

sentiments = []

def analysis(review: str):
    result = sent_pl(review[:512])[0]
    neutral_threshold = 0.6 
    
    positive_score = result['score'] if result['label'] == 'POSITIVE' else 0
    negative_score = result['score'] if result['label'] == 'NEGATIVE' else 0
    
    if positive_score < neutral_threshold and negative_score < neutral_threshold:
        return 'NEUTRAL', 0.0 

    return result['label'], result['score']

def graph(sentiments: list):
    labels = ['POSITIVE', 'NEGATIVE', 'NEUTRAL']
    pos_count = sentiments.count('POSITIVE')
    neg_count = sentiments.count('NEGATIVE')
    neutral_count = sentiments.count('NEUTRAL')
    
    sizes = [pos_count, neg_count, neutral_count]
    colors = ['green', 'red', 'gray']  

    plt.figure(figsize=(6, 6))
    plt.pie(
        sizes,
        labels=labels,
        autopct=lambda p: f'{p:.1f}%' if p > 0 else '',
        startangle=140,
        colors=colors
    )
    plt.axis('equal')
    plt.title("Sentiment Analysis Results")

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return buf

@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    global sentiments
    try:
        # Read the CSV file
        df = pd.read_csv(file.file)
        
        # Ensure 'reviewText' column has no NaN or float values
        df['reviewText'] = df['reviewText'].fillna("")  
        df['reviewText'] = df['reviewText'].apply(str)  

        all_reviews = df['reviewText']
        reviews = all_reviews[:100]
        for review in reviews:
            sentiment, score = analysis(review)
            sentiments.append(sentiment)
        
        sentiment_counts = dict(pd.Series(sentiments).value_counts())
        
        # Convert the counts' values to int (to make them JSON serializable)
        sentiment_counts = {k: int(v) for k, v in sentiment_counts.items()}
        
        return JSONResponse(content={"message": "Sentiments processed successfully!", "counts": sentiment_counts})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/graph")
def get_graph():
    global sentiments
    if not sentiments:
        raise HTTPException(status_code=400, detail="No sentiments available to generate a graph.")

    graph_buf = graph(sentiments)
    return StreamingResponse(graph_buf, media_type="image/png")
