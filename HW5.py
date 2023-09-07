import os

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Список класів активностей
classes = ["idle", "running", "stairs", "walking"]

# Порожній DataFrame для об'єднання всіх даних
all_data = pd.DataFrame()

# Проходимо по кожному класу
for activity in classes:
    # Визначаемо шлях до папки з відповідним класом
    folder_path = os.path.join(".", activity)

    # Перевірка наявності папки перед завантаженням
    if os.path.exists(folder_path):
        # Проходимо по всім файлам в папці
        for filename in os.listdir(folder_path):
            if filename.endswith(".csv"):
                # Зчитуємо дані з CSV файлу
                file_path = os.path.join(folder_path, filename)
                data = pd.read_csv(file_path)

                # Додаемо стовпець activity з ім'ям класу
                data['activity'] = activity

                # Об'єднуемо дані з іншими даними
                all_data = pd.concat([all_data, data], ignore_index=True)
    else:
        print(f"Папка {folder_path} для класу {activity} не існує.")

# Розділення даних на навчальний та тестовий набори
train_data, test_data = train_test_split(all_data, test_size=0.2, random_state=42)

# Розрахунок часових ознак
time_features = ['mean', 'std', 'median']

features = []
for axis in ['accelerometer_X', 'accelerometer_Y', 'accelerometer_Z']:
    for stat in time_features:
        feature_name = f"{axis}_{stat}"
        features.append(feature_name)
        train_data[feature_name] = train_data.groupby('activity')[axis].transform(stat)
        test_data[feature_name] = test_data.groupby('activity')[axis].transform(stat)


# Вибір фіч для навчання та тестування
X_train = train_data[features]
y_train = train_data['activity']
X_test = test_data[features]
y_test = test_data['activity']

# # Навчання моделі SVM
# svm_model = SVC()
# svm_model.fit(X_train, y_train)

# Навчання моделі SVM з підтримкою ймовірностей
svm_model = SVC(probability=True)
svm_model.fit(X_train, y_train)

# Прогноз ймовірностей на тестовому наборі
svm_probabilities = svm_model.predict_proba(X_test)

# Обчислення ROC AUC для моделі SVM
svm_roc_auc = roc_auc_score(y_test, svm_probabilities, multi_class='ovr')
print(f"ROC AUC for SVM: {svm_roc_auc}")

# Прогноз на тестовому наборі та оцінка моделі SVM
svm_predictions = svm_model.predict(X_test)
svm_accuracy = accuracy_score(y_test, svm_predictions)
print("Точність моделі SVM:", svm_accuracy)

# Навчання моделі Random Forest
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

# Прогноз на тестовому наборі та оцінка моделі Random Forest
rf_predictions = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_predictions)
print("Точність моделі Random Forest:", rf_accuracy)

# Порівняння результатів моделей

# Модель SVM
print("Модель SVM:")
svm_confusion = confusion_matrix(y_test, svm_predictions)
print("Матриця плутанини:")
print(svm_confusion)

print("\nЗвіт про класифікації:")
svm_classification_report = classification_report(y_test, svm_predictions)
print(svm_classification_report)

rf_roc_auc = roc_auc_score(y_test, rf_model.predict_proba(X_test), multi_class='ovr')
print(f"ROC AUC: {rf_roc_auc}")


# svm_roc_auc = roc_auc_score(y_test, svm_model.decision_function(X_test), multi_class='ovr')
# print(f"ROC AUC: {svm_roc_auc}")


# Модель Random Forest
print("\nМодель Random Forest:")
rf_confusion = confusion_matrix(y_test, rf_predictions)
print("Матриця плутанини:")
print(rf_confusion)

print("\nЗвіт про класифікації:")
rf_classification_report = classification_report(y_test, rf_predictions)
print(rf_classification_report)

rf_roc_auc = roc_auc_score(y_test, rf_model.predict_proba(X_test), multi_class='ovr')
print(f"ROC AUC: {rf_roc_auc}")

# Тюнінг гіперпараметрів для SVM
param_grid_svm = {'C': [0.1, 1, 10],
                  'kernel': ['linear', 'rbf', 'poly', 'sigmoid']}
svm_grid = GridSearchCV(SVC(), param_grid_svm, cv=3)
svm_grid.fit(X_train, y_train)

# Найкращі гіперпараметри для SVM
best_params_svm = svm_grid.best_params_
print("\nНайкращі гіперпараметри для SVM:")
print(best_params_svm)

# Тюнінг гіперпараметрів для Random Forest
param_grid_rf = {'n_estimators': [50, 100, 200],
                 'max_depth': [None, 10, 20, 30]}
rf_grid = GridSearchCV(RandomForestClassifier(), param_grid_rf, cv=3)
rf_grid.fit(X_train, y_train)

# Найкращі гіперпараметри для Random Forest
best_params_rf = rf_grid.best_params_
print("\nНайкращі гіперпараметри для Random Forest:")
print(best_params_rf)

# Метрика точності (Precision)
precision = precision_score(y_test, svm_predictions, average='weighted')
print("Точність (Precision) моделі SVM:", precision)

# Метрика відгук (Recall)
recall = recall_score(y_test, svm_predictions, average='weighted')
print("Відгук (Recall) моделі SVM:", recall)

# F1-мера
f1 = f1_score(y_test, svm_predictions, average='weighted')
print("F1-мера моделі SVM:", f1)

# Метрика точності (Precision)
precision_rf = precision_score(y_test, rf_predictions, average='weighted')
print("Точність (Precision) моделі Random Forest:", precision_rf)

# Метрика відгук (Recall)
recall_rf = recall_score(y_test, rf_predictions, average='weighted')
print("Відгук (Recall) моделі Random Forest:", recall_rf)

# F1-мера
f1_rf = f1_score(y_test, rf_predictions, average='weighted')
print("F1-мера моделі Random Forest:", f1_rf)

# Матриця плутанини для моделі SVM
svm_confusion = confusion_matrix(y_test, svm_predictions)
print("Матриця плутанини для моделі SVM:")
print(svm_confusion)

# Матриця плутанини для моделі Random Forest
rf_confusion = confusion_matrix(y_test, rf_predictions)
print("Матриця плутанини для моделі Random Forest:")
print(rf_confusion)

# Визначення важливості функцій
feature_importances = rf_model.feature_importances_
sorted_idx = feature_importances.argsort()[::-1]

# Виведення найважливіших функцій
for i, idx in enumerate(sorted_idx[:10]):
    print(f"{i+1}. {features[idx]}: {feature_importances[idx]}")
