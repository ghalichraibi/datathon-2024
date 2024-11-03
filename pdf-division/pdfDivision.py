import fitz 
import re
import boto3
import datetime
import os
from dotenv import load_dotenv
financial_keywords = r"\b(Financial Highlights|Summary of Consolidated Quarterly Results|Major Cash Flow Components)\b"
financial_data_pattern = r"\b\d{1,3}(?:[ ,]\d{3})*(?:\.\d{1,2})?(?:[%]|(?:\$|€|¥|£)?)\b"

def delete_file(pdf_file_path):
    # Check if the file exists
    if os.path.exists(pdf_file_path):
        # Delete the file
        os.remove(pdf_file_path)
        print(f"{pdf_file_path} has been deleted.")
    else:
        print(f"{pdf_file_path} does not exist.")

def extract_financial_pages(pdf_path, output_name):
    doc = fitz.open(pdf_path)
    new_doc = fitz.open()

    has_page_in_new_doc = 0

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        
        if len(re.findall(financial_data_pattern, text)) > 30:
            has_page_in_new_doc += 1
            new_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)

    current_datetime = datetime.datetime.now() 
    output_id = current_datetime.strftime("%Y%m%d_%H%M%S")

    output_pdf_path = f"{output_name}_{output_id}.pdf"

    new_doc.save(output_pdf_path)
    new_doc.close()
    
    if has_page_in_new_doc:
        print(f"Financial data extracted to {output_pdf_path}")
        drop_pdf_scraped_in_s3(output_pdf_path)

    else:
        print("No financial data found in the PDF.")

    delete_file(output_pdf_path)

    doc.close()

def drop_pdf_scraped_in_s3(file_path):

    # Nom du bucket et de l'emplacement désiré du fichier
    s3_key = file_path

    # Télécharger le fichier sur le bucket S3
    s3_client.upload_file(file_path, bucket_name, s3_key)

    print(f'File {file_path} uploaded to {bucket_name}/{s3_key}')



load_dotenv()
# Créer une session boto3
bucket_name = os.getenv("BUCKET_NAME")

session = boto3.Session(
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("REGION_NAME"),
)

# Créer un client S3
s3_client = session.client('s3')

# relatif a la knowledgebase
bedrock_client = boto3.client('bedrock-agent')

# Relatif a l'agent
my_prompt = boto3.client('bedrock-agent-runtime')

my_file_to_scrap = "CP_AnnualReport2020_SECURED.pdf"
file_scraped_name = f"financial_file_scraped"

# extract_financial_pages(my_file_to_scrap, file_scraped_name) 

# Mise a jour de KB
# bedrock_client.start_ingestion_job(dataSourceId = "BASBLNN6II", knowledgeBaseId = "ISOUIL1PJ9")

# Ask
response = my_prompt.invoke_agent(
    agentId = "W4XOVSG5PL",
    agentAliasId = "IW4COIFE6B",
    sessionId =  datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
    inputText = f"""Fill the following json template with values found in the company financial report named {my_file_to_scrap} in your knowledge base. Your response should not contain anything else than the filled out template. If you cannot find a value, put a null value in the appropriate field. For the summary field, write a quick summary of the company's activity sector based on its name. Json template: 
    'Company Name': ,
    'Fiscal Year': ,
    'Report Date': ,
    'Currency': ,
    'Summary': ,
    'Total earnings':,
    'Total Net Income': ,
    'Total Operating Income': ,
    'Total Expenses': ,
    'Cost of Goods Sold (COGS)': ,
    'Selling, General, and Administrative (SG&A)': ,
    'Research and Development (R&D)': ,
    'Depreciation and Amortization': ,
    'Interest Expense': ,
    'Other Expenses': ,
    'Total Debt': ,
    'Debt-to-Equity Ratio': ,
    'Long-Term Debt': ,
    'Short-Term Debt': ,
    'Total Equity': ,
    'Gross Profit Margin': ,
    'Operating Profit Margin': ,
    'Net Profit Margin': ,
    'Return on Assets (ROA)': ,
    'Return on Equity (ROE)': ,
""")
completion = ""
for event in response.get("completion"):
    chunk = event["chunk"]
    completion += chunk["bytes"].decode()
print(completion)