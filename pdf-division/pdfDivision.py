import fitz 
import re

financial_keywords = r"\b(Financial Highlights|Summary of Consolidated Quarterly Results|Major Cash Flow Components)\b"
financial_data_pattern = r"\$\s?\d+(,\d{3})*(\.\d+)?|\d{1,3}(,\d{3})+"

def extract_financial_pages(pdf_path, output_pdf_path):
    doc = fitz.open(pdf_path)
    financial_pages = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        
        if re.search(financial_data_pattern, text):
            financial_pages.append(page_num) 

    
    if financial_pages:
        new_doc = fitz.open()
        for page_num in financial_pages:
            new_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)
        
        new_doc.save(output_pdf_path)
        new_doc.close()
        print(f"Financial data extracted to {output_pdf_path}")
    else:
        print("No financial data found in the PDF.")

    doc.close()

extract_financial_pages("doc-financier-cn-2023.pdf", "financial_data_scraped.pdf")