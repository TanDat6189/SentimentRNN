import torch
from nltk.tokenize import word_tokenize
import json
from model import RNNModel

from data import vocab

# Khởi tạo lại mô hình với đúng cấu hình như lúc train
model = RNNModel(vocab_size=len(vocab), embedding_dim=100, hidden_dim=128, output_dim=3, pretrained=True)
model.load_state_dict(torch.load('my_model_pretrained_True.pt'))
model.eval()

def to_indices(tokens, vocab, max_len=50):
    idxs = [vocab.get(t, 1) for t in tokens][:max_len]
    return idxs + [0] * (max_len - len(idxs))

def predict_sentiment(text):
    tokens = word_tokenize(text.lower())
    indices = to_indices(tokens, vocab)
    input_tensor = torch.tensor(indices).unsqueeze(0)  # Batch size = 1
    with torch.no_grad():
        output = model(input_tensor)
        pred = torch.argmax(output, dim=1).item()
        label_map = {0: "Positive", 1: "Negative", 2: "Neutral"}
        return label_map[pred]


# print(predict_sentiment("Tôi cảm thấy rất tự tin khi công việc của mình được đánh giá cao."))
# print(predict_sentiment("Tôi cảm thấy áp lực khi công việc dồn lại trong thời gian ngắn."))
# print(predict_sentiment("Tôi đang chuẩn bị báo cáo cho hội thảo khoa học sắp tới."))

# print(predict_sentiment("I have no complaints or praise."))
# print(predict_sentiment("I can’t focus no matter how hard I try."))
# print(predict_sentiment("I feel genuinely satisfied with my learning."))
# print(predict_sentiment("I love you"))

print(predict_sentiment("I finally understood the math concept today!"))  # Positive
print(predict_sentiment("I forgot to bring my notes to class."))  # Negative
print(predict_sentiment("My meeting ended earlier than expected."))  # Neutral
print(predict_sentiment("The group project is progressing well."))  # Positive
print(predict_sentiment("I missed an important deadline."))  # Negative
print(predict_sentiment("I learned something new in the seminar."))  # Positive
print(predict_sentiment("My internet kept disconnecting during the lecture."))  # Negative
print(predict_sentiment("I'm feeling indifferent about my progress."))  # Neutral
print(predict_sentiment("The meeting was okay, not bad, not good."))  # Neutral
print(predict_sentiment("Today felt like a regular workday."))  # Neutral
print(predict_sentiment("I'm feeling neutral about the class material."))  # Neutral
print(predict_sentiment("I didn’t make much progress today."))  # Neutral
print(predict_sentiment("My day was a mix of good and bad."))  # Neutral
print(predict_sentiment("I'm staying on track, but it feels like a routine."))  # Neutral
print(predict_sentiment("I'm not sure if I’m doing better or worse."))  # Neutral
print(predict_sentiment("I finished my tasks, but I’m not excited."))  # Neutral
