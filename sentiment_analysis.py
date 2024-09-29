import pandas as pd
import matplotlib.pyplot as plt
from langchain.prompts import PromptTemplate
from transformers import pipeline

model = "distilbert-base-uncased-finetuned-sst-2-english" 
sent_pl = pipeline("sentiment-analysis", model=model)

prompt_temp=PromptTemplate(input_variables=["reviewText"], template="Analyze the sentiment of the following review: {reviewText}")

def analysis(review: str):
    result=sent_pl(review[:512])[0]
    return result['label'], result['score']


def graph(sentiments: list):
    labels = ['POSITIVE', 'NEGATIVE']
    pos_count = sentiments.count('POSITIVE')
    neg_count = sentiments.count('NEGATIVE')
    sizes = [pos_count, neg_count]
    if sum(sizes) == 0:
        print("No sentiments to display.")
        return
    colors = ['green', 'red']
    plt.figure(figsize=(6, 6))
    plt.pie(sizes, labels=labels, autopct=lambda p: '{:.1f}%'.format(p) if p > 0 else '', startangle=140, colors=colors)
    plt.axis('equal')
    plt.title("Sentiment Analysis Results")
    plt.show()

def main(csv_path):
    df=pd.read_csv(csv_path)
    if 'reviewText' not in df.columns:
        print("Error: the csv file must contain a column called reviewText")
        return 
    sentiments=[]
    for review in df['reviewText']:
        if isinstance(review, str) and review:
            sentiment, score = analysis(review)
            sentiments.append(sentiment)
        else:
            print(f"Skippingig Invalid reviews: {review}")
    graph(sentiments)

if __name__=="__main__":
    csv_path=input("Enter the path of the CSV file: ")
    main(csv_path)
