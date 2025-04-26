import pandas as pd

# Đọc dữ liệu
df = pd.read_csv("datasetVietnamese.csv")

# 1. Kiểm tra cấu trúc cột
expected_columns = ['text', 'label']
if list(df.columns) != expected_columns:
    raise ValueError(f"Cột không đúng, mong đợi: {expected_columns}, thực tế: {list(df.columns)}")

# 2. Kiểm tra dữ liệu trống
missing = df.isnull().sum()
if missing.any():
    print("\n⚠️ Dữ liệu thiếu:")
    print(missing[missing > 0])
else:
    print("\n✅ Không có ô trống.")

# 3. Kiểm tra nhãn hợp lệ
valid_labels = {'Positive', 'Negative', 'Neutral'}
invalid_labels = df[~df['label'].isin(valid_labels)]
if not invalid_labels.empty:
    print("\n⚠️ Nhãn không hợp lệ:")
    print(invalid_labels)
else:
    print("\n✅ Tất cả nhãn đều hợp lệ.")

# 4. Kiểm tra độ dài văn bản < 50 từ
df['word_count'] = df['text'].astype(str).apply(lambda x: len(x.split()))
long_texts = df[df['word_count'] >= 50]
if not long_texts.empty:
    print(f"\n⚠️ {len(long_texts)} văn bản vượt quá 50 từ.")
else:
    print("\n✅ Tất cả văn bản đều dưới 50 từ.")

# 5. Kiểm tra và xóa dòng trùng lặp
duplicates = df.duplicated()
if duplicates.any():
    print(f"\n⚠️ Có {duplicates.sum()} dòng trùng lặp. Đang xóa...")
    df = df.drop_duplicates()
    print("✅ Đã xóa dòng trùng lặp.")
else:
    print("\n✅ Không có dòng trùng lặp.")

# 6. Thống kê phân bố nhãn
print("\n📊 Phân bố nhãn:")
label_counts = df['label'].value_counts(normalize=True) * 100
print(label_counts.round(2).astype(str) + '%')

# Lưu file đã được làm sạch
output_path = "sentiment_data.csv"
df.drop(columns=['word_count'], inplace=True)  # Xóa cột phụ trợ nếu không cần
df.to_csv(output_path, index=False)
print(f"\n✅ Dữ liệu sạch đã được lưu tại: {output_path}")