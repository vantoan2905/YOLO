from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Giả sử bạn có danh sách các nhãn thực tế (ground truth) và nhãn dự đoán
y_true = [0, 1, 2, 2, 0, 1]  # Ví dụ về nhãn thực tế
y_pred = [0, 2, 2, 2, 0, 0]  # Ví dụ về nhãn dự đoán

# Tạo Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

# Liệt kê các lớp trong mô hình của bạn
classes = ['class_0', 'class_1', 'class_2']  # Ví dụ tên các lớp

# Trực quan hóa Confusion Matrix bằng seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
