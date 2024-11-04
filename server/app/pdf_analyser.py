import pymupdf 
import re
import boto3
import datetime
import os
import json
import time
from dotenv import load_dotenv
from botocore.exceptions import ClientError

financial_data_pattern = r"\b\d{1,3}(?:[ ,]\d{3})*(?:\.\d{1,2})?(?:[%]|(?:\$|€|¥|£)?)\b"

bucket_name = ""
s3_client = None
bedrock_client = None
my_prompt = None

def init():
    global bucket_name, s3_client, bedrock_client, my_prompt
    
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

    # Relatif a la KnowledgeBase
    bedrock_client = boto3.client('bedrock-agent',region_name=os.getenv("REGION_NAME"))

    # Relatif a l'agent
    my_prompt = boto3.client('bedrock-agent-runtime',region_name=os.getenv("REGION_NAME"))

def check_ingestion_job_status(job_id):
    try:
        response = bedrock_client.get_ingestion_job(dataSourceId = "BASBLNN6II", 
                                                    knowledgeBaseId = "ISOUIL1PJ9",
                                                    ingestionJobId=job_id)
        status = response['ingestionJob']['status']
        return status
    except ClientError as e:
        print(f"An error occurred: {e}")
        return None

def pdf_to_json(pdf_path):

    current_datetime = datetime.datetime.now() 
    output_id = current_datetime.strftime("%Y%m%d_%H%M%S")
    output_path_pdf = pdf_path.replace(".pdf", f"_{output_id}.pdf")
    extract_financial_pages(pdf_path, output_path_pdf) 

    # Mise a jour de KB
    ingestion_job_response = bedrock_client.start_ingestion_job(dataSourceId = "BASBLNN6II", knowledgeBaseId = "ISOUIL1PJ9")
    ingestion_job_id = ingestion_job_response['ingestionJob']['ingestionJobId']
    # Attendre que la tâche soit complétée
    status = check_ingestion_job_status(ingestion_job_id)
    while status not in ["COMPLETE", "FAILED"]:
        print(f"Current status of ingestion job {ingestion_job_id}: {status}")
        time.sleep(10)  # Attendre 60 secondes avant de vérifier à nouveau
        status = check_ingestion_job_status(ingestion_job_id)

    # Afficher le statut final
    if status == "COMPLETE":
        print(f"Knowledge base has been updated successfully. Final status of ingestion job {ingestion_job_id}: {status}")

        # Ask
        response = my_prompt.invoke_agent(
            agentId = "W4XOVSG5PL",
            agentAliasId = "IW4COIFE6B",
            sessionId =  datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
            inputText = f"""Fill the following json template with values found in the company financial report named {output_path_pdf} in your knowledge base. 
            Your response should not contain anything else than the filled out template.
            For the summary field, write a quick summary of the company's activity sector based on its name. For the numbers, don't use commas for thousands separators (use nothing).
            Every field should have a value, avoid writing null. Do not forget the curly braces. 
            Json template: 
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
        """
        )
    completion = ""
    for event in response.get("completion"):
        chunk = event["chunk"]
        completion += chunk["bytes"].decode()

    # Make JSON-compatible (use double quotes for keys and null for missing values)
    json_compatible_completion = completion.replace("'", '"')
    json_compatible_completion = json_compatible_completion.replace(": ,", ": null,")
    json_compatible_completion = json_compatible_completion.replace(":,", ": null,")
    json_compatible_completion = re.sub(r":\s*}", ": null}", json_compatible_completion)

    # Convert to a dictionary
    try:
        result = json.loads(json_compatible_completion)
    except json.JSONDecodeError as e:
        print("Error decoding JSON:", e)
        result = None

    print("JSON completion:", completion)
    print("JSON-compatible completion:", json_compatible_completion)
    print("Dictionary completion:", result)
    return result

def extract_financial_pages(pdf_path, output_pdf_path):
    doc = pymupdf.open(pdf_path)
    new_doc = pymupdf.open()

    has_page_in_new_doc = 0

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        
        if len(re.findall(financial_data_pattern, text)) > 20:
            has_page_in_new_doc += 1
            new_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)

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
    extracted_name = os.path.basename(file_path)

    # Nom du bucket et de l'emplacement désiré du fichier
    s3_key = extracted_name

    # Télécharger le fichier sur le bucket S3
    s3_client.upload_file(file_path, bucket_name, s3_key)

    print(f'File {file_path} uploaded to {bucket_name}/{s3_key}')

def delete_file(pdf_file_path):
    # Check if the file exists
    if os.path.exists(pdf_file_path):
        # Delete the file
        os.remove(pdf_file_path)
        print(f"{pdf_file_path} has been deleted.")
    else:
        print(f"{pdf_file_path} does not exist.")

def delete_from_s3(pdf_file_path):
    # Vérifier si l'objet existe 
    try: 
        s3_client.head_object(Bucket=bucket_name, Key=pdf_file_path) 
        object_exists = True 
        print(f'The object {pdf_file_path} exists in bucket {bucket_name}.') 
    except s3_client.exceptions.ClientError as e: 
        if e.response['Error']['Code'] == '404': 
            object_exists = False 
            print(f'The object {pdf_file_path} does not exist in bucket {bucket_name}.') 
        else: 
            raise

    # Supprimer l'objet si il existe 
    if object_exists: 
        s3_client.delete_object(Bucket=bucket_name, Key=pdf_file_path) 
        print(f'File {pdf_file_path} deleted from bucket {bucket_name}.')


init()

if __name__ == "__main__":
    pdf_to_json("test.pdf")
