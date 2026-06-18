import os
import pytesseract
import cv2
import Levenshtein

class OCRTester:
    def __init__(self, tesseract_cmd_path=None):
        if tesseract_cmd_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd_path
        self.results = []

    def calculate_cer(self, reference_text, recognized_text):
        """Вычисляет Character Error Rate (CER)"""
        ref_clean = " ".join(reference_text.split()).lower()
        rec_clean = " ".join(recognized_text.split()).lower()

        if len(ref_clean) == 0:
            return 0.0

        distance = Levenshtein.distance(ref_clean, rec_clean)
        cer = (distance / len(ref_clean)) * 100.0
        return cer

    def calculate_wer(self, reference_text, recognized_text):
        """Вычисляет Word Error Rate (WER)"""
        ref_words = reference_text.split()
        rec_words = recognized_text.split()

        if len(ref_words) == 0:
            return 0.0

        distance = Levenshtein.distance(ref_words, rec_words)
        wer = (distance / len(ref_words)) * 100.0
        return wer

    def calculate_accuracy(self, reference_text, recognized_text):
        """Вычисляет точность на основе CER"""
        cer = self.calculate_cer(reference_text, recognized_text)
        accuracy = max(0.0, 100.0 - cer)
        return accuracy

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

                    cer = self.calculate_cer(reference_text, recognized_text)
                    accuracy = self.calculate_accuracy(reference_text, recognized_text)

                    self.results.append({
                        'file': filename,
                        'cer': round(cer, 2),
                        'accuracy': round(accuracy, 2)
                    })
                    print(f"Файл: {filename} | CER: {cer:.2f}% | Точность: {accuracy:.2f}%")

                except Exception as e:
                    print(f"Ошибка при обработке файла {filename}: {str(e)}")

        return self.results

    def generate_report(self):
        if not self.results:
            return "Нет данных для формирования отчета."

        avg_cer = sum(item['cer'] for item in self.results) / len(self.results)
        avg_accuracy = sum(item['accuracy'] for item in self.results) / len(self.results)

        report = "--- ИТОГОВЫЙ ОТЧЕТ ---\n"
        report += f"Обработано файлов: {len(self.results)}\n"
        report += f"Средний CER: {avg_cer:.2f}%\n"
        report += f"Средняя точность: {avg_accuracy:.2f}%\n"
        return report


# Пример использования
if __name__ == "__main__":
    tester = OCRTester(tesseract_cmd_path=r'D:\tesseract\tesseract.exe')
    
    results = tester.run_batch_test(
        images_dir="test_images/",
        reference_file="reference.txt"
    )
    
    report = tester.generate_report()
    print(report)