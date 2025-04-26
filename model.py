import torch
import torch.nn as nn
import torchtext.vocab as vocab
from data import vocab as my_vocab

class RNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, pretrained=False):
        super().__init__()
        # Khởi tạo embedding layer (dùng GloVe nếu pretrained=True)
        # [Dùng nn.Embedding, xử lý pretrained với GloVe]
        # Embedding Layer
        if pretrained:
            # Tải GloVe vector (100d)
            glove = vocab.GloVe(name='6B', dim=embedding_dim)
            # Khởi tạo ma trận embedding (vocab_size x embedding_dim)
            embedding_weights = torch.zeros(vocab_size, embedding_dim)

            for word, idx in my_vocab.items():
                if idx >= vocab_size:
                    continue
                try:
                    embedding_weights[idx] = glove[word]
                except KeyError:
                    # Nếu từ không có trong GloVe thì khởi tạo ngẫu nhiên
                    embedding_weights[idx] = torch.randn(embedding_dim) * 0.1

            self.embedding = nn.Embedding.from_pretrained(embedding_weights, freeze=False, padding_idx=0)
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # Khởi tạo khối RNN layer
        # [Dùng nn.RNN với batch_first=True]
        self.rnn = nn.RNN(input_size=embedding_dim, hidden_size=hidden_dim, batch_first=True)

        # Khởi tạo tầng Dense để dự đoán 3 nhãn
        # [Dùng nn.Linear, nhận hidden state từ RNN]
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        # text: (batch_size, seq_len)

        # Chuyển text thành embedding
        embedded = self.embedding(text)         # (batch_size, seq_len, embedding_dim)

        # Đưa qua khối RNN để lấy hidden state cuối
        output, hidden = self.rnn(embedded)     # hidden: (1, batch_size, hidden_dim
        # Lấy hidden state cuối cùng
        last_hidden = hidden.squeeze(0)         # (batch_size, hidden_dim)

        # Đưa hidden state qua tầng Dense để dự đoán 3 nhãn
        out = self.fc(last_hidden)              # (batch_size, output_dim)

        # [Trả về kết quả dự đoán]
        return out

#Khởi tạo mô hình
# model = RNNModel(vocab_size=vocab_size, embedding_dim=100, hidden_dim=128, output_dim=3, pretrained=True)
