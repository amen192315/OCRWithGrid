import os
import pytesseract
import cv2
from difflib import SequenceMatcher

class OCRTester:
    def __init__(self, tesseract_cmd_path=None):
        if tesseract_cmd_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd_path
        self.results = []

    def calculate_similarity(self, reference_text, recognized_text):
        ref_clean = " ".join(reference_text.split()).lower()
        rec_clean = " ".join(recognized_text.split()).lower()
        
        matcher = SequenceMatcher(None, ref_clean, rec_clean)
        return matcher.ratio() * 100.0

    def run_batch_test(self, images_dir, reference_file, psm_mode=3):
        print(f"Начало тестирования директории: {images_dir}")
        custom_config = f'--oem 3 --psm {psm_mode} -l rus+eng'
        
        with open(reference_file, 'r', encoding='utf-8') as f:
            reference_text = f.read()

        for filename in os.listdir(images_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(images_dir, filename)
                try:
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        continue
                        
                    recognized_text = pytesseract.image_to_string(img, config=custom_config)
                    
                    accuracy = self.calculate_similarity(reference_text, recognized_text)
                    
                    self.results.append({
                        'file': filename,
                        'accuracy': round(accuracy, 2)
                    })
                    print(f"Файл: {filename} | Точность: {accuracy:.2f}%")
                    
                except Exception as e:
                    print(f"Ошибка при обработке файла {filename}: {str(e)}")
                    
        return self.results

    def generate_report(self):
        if not self.results:
            return "Нет данных для формирования отчета."
            
        avg_accuracy = sum(item['accuracy'] for item in self.results) / len(self.results)
        report = f"--- ИТОГОВЫЙ ОТЧЕТ ---\n"
        report += f"Обраработано файлов: {len(self.results)}\n"
        report += f"Средняя точность распознавания: {avg_accuracy:.2f}%\n"
        return report

# Листинг X - Модуль автоматизированной оценки точности распознавания текста