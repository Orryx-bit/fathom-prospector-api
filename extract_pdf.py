
import pypdf
import sys

def extract_text(pdf_path, output_path):
    try:
        reader = pypdf.PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n\n"
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Successfully extracted text to {output_path}")
    except Exception as e:
        print(f"Error extracting text: {e}")
        sys.exit(1)

if __name__ == "__main__":
    extract_text("Bellafill_Practice_Fit_Analysis.pdf", "bellafill_analysis.txt")
