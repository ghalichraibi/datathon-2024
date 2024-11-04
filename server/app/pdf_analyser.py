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

    bucket_name = 'scraped-pdf-bucket'
    if not bucket_name:
        print("Error: BUCKET_NAME is not set in environment variables.")
        return
    
    bucket_name = os.getenv("BUCKET_NAME")
    aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    print(f"Bucket Name: {bucket_name}")
    print(f"AWS Access Key: {'Loaded' if aws_access_key else 'Not loaded'}")
    print(f"AWS Secret Key: {'Loaded' if aws_secret_key else 'Not loaded'}")

    if not aws_access_key or not aws_secret_key:
        print("Error: AWS credentials are not set in the environment variables.")
        return

    session = boto3.Session(
        aws_access_key_id='AKIA5HE2WWPH2CRMMQMY',
        aws_secret_access_key='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',
        region_name='us-west-2',
    )

    s3_client = session.client('s3')
    bedrock_client = boto3.client('bedrock-agent', region_name='us-west-2')
    my_prompt = boto3.client('bedrock-agent-runtime', region_name='us-west-2')
    print("Initialization complete. S3 and Bedrock clients are set up.")

def check_ingestion_job_status(job_id):
    try:
        response = bedrock_client.get_ingestion_job(
            dataSourceId="BASBLNN6II", 
            knowledgeBaseId="ISOUIL1PJ9",
            ingestionJobId=job_id
        )
        # Check if response is a dictionary
        if isinstance(response, dict):
            status = response.get('ingestionJob', {}).get('status')
            print(f"Ingestion job {job_id} status: {status}")
            return status
        else:
            # Log tuple response as error
            print(f"Error: Unexpected tuple response format: {response}")
            return None
    except ClientError as e:
        print(f"An error occurred while checking ingestion job status: {e}")
        return None


    
def pdf_to_json(pdf_path):
    if not pdf_path:
        print("Error: pdf_path is None")
        return {"error": "No PDF path provided"}, 400

    current_datetime = datetime.datetime.now() 
    output_id = current_datetime.strftime("%Y%m%d_%H%M%S")
    output_path_pdf = pdf_path.replace(".pdf", f"_{output_id}.pdf")
    
    try:
        # Attempt to extract financial pages
        extract_financial_pages(pdf_path, output_path_pdf)
    except Exception as e:
        print(f"Error extracting financial pages: {e}")
        return {"error": "Failed to extract financial pages"}, 500

    try:
        # Start the ingestion job
        ingestion_job_response = bedrock_client.start_ingestion_job(
            dataSourceId="BASBLNN6II", 
            knowledgeBaseId="ISOUIL1PJ9"
        )
        # Ensure ingestion job response is a dictionary
        if isinstance(ingestion_job_response, dict):
            ingestion_job_id = ingestion_job_response.get('ingestionJob', {}).get('ingestionJobId')
            print(f"Started ingestion job {ingestion_job_id}")
        else:
            print("Error: Unexpected tuple or non-dict response:", ingestion_job_response)
            return {"error": "Invalid response from ingestion job"}, 500
    except ClientError as e:
        print(f"Failed to start ingestion job: {e}")
        return {"error": "Failed to start ingestion job"}, 500

    # Polling for job completion
    status = check_ingestion_job_status(ingestion_job_id)
    while status not in ["COMPLETE", "FAILED"]:
        print(f"Current status of ingestion job {ingestion_job_id}: {status}")
        time.sleep(10)
        status = check_ingestion_job_status(ingestion_job_id)

    if status == "COMPLETE":
        print(f"Ingestion job {ingestion_job_id} completed successfully.")
        try:
            # Invoke the Bedrock agent with a detailed prompt
            response = my_prompt.invoke_agent(
                agentId="W4XOVSG5PL",
                agentAliasId="CIGVELRLL1",
                sessionId=datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
                inputText=(
                    f"Please fill out the following JSON template using values from the financial report located at '{output_path_pdf}'. "
                    "Each field must have a value. If the exact value is not found in the report, estimate based on general knowledge or indicate 'N/A'. "
                    "Do not add any extra fields, comments, or explanations. Ensure the JSON is structured exactly as below with double quotes for each key and value. "
                    "Return the JSON in the specified format without any surrounding text, headers, or footers:\n\n"
                    "{\n"
                    '    "Company Name": "PrivateTech",\n'
                    '    "Fiscal Year": 2022,\n'
                    '    "Report Date": "2022-12-31",\n'
                    '    "Currency": "USD",\n'
                    '    "Summary": "PrivateTech develops software solutions for e-commerce optimization in North America.",\n'
                    '    "Total earnings": 5006780,\n'
                    '    "Total Net Income": 707800,\n'
                    '    "Total Operating Income": 609800,\n'
                    '    "Total Expenses": 4309700,\n'
                    '    "Cost of Goods Sold (COGS)": 2500560,\n'
                    '    "Selling, General, and Administrative (SG&A)": 1225000,\n'
                    '    "Research and Development (R&D)": 400180,\n'
                    '    "Depreciation and Amortization": 200000,\n'
                    '    "Interest Expense": 100000,\n'
                    '    "Other Expenses": "N/A",\n'
                    '    "Total Debt": 2000000,\n'
                    '    "Debt-to-Equity Ratio": 0.5,\n'
                    '    "Long-Term Debt": 1500000,\n'
                    '    "Short-Term Debt": 500000,\n'
                    '    "Total Equity": 4000000,\n'
                    '    "Gross Profit Margin": 0.55,\n'
                    '    "Operating Profit Margin": 0.12,\n'
                    '    "Net Profit Margin": 0.14,\n'
                    '    "Return on Assets (ROA)": 0.10,\n'
                    '    "Return on Equity (ROE)": 0.175\n'
                    "}"
                ))

            print("Raw response from invoke_agent:", response)
            if isinstance(response, tuple):
                print("Error: Received unexpected tuple from invoke_agent:", response)
                return {"error": "Invalid response from agent"}, 500
        except ClientError as e:
            print(f"Failed to invoke agent: {e}")
            return {"error": "Agent invocation failed"}, 500
    else:
        print(f"Ingestion job {ingestion_job_id} failed.")
        return {"error": "Ingestion job failed"}, 500

    completion = ""
    # Process the response to handle tuple anomalies
    for event in response.get("completion", []):
        chunk = event.get("chunk", {})
        if isinstance(chunk, tuple):
            print("Error: Unexpected tuple in completion:", chunk)
        else:
            completion += chunk.get("bytes", b"").decode("utf-8")

    print("Raw completion content before JSON parsing:", completion)

    try:
        json_compatible_completion = completion.replace("'", '"')
        json_compatible_completion = re.sub(r":\s*}", ": null}", json_compatible_completion)
        print("JSON-compatible completion string:", json_compatible_completion)
        result = json.loads(json_compatible_completion)
        print("Parsed JSON:", result)
        return result
    except json.JSONDecodeError as e:
        print("Error decoding JSON:", e)
        return {"error": "Failed to decode JSON"}, 500



def extract_financial_pages(pdf_path, output_pdf_path):
    if not pdf_path or not output_pdf_path:
        print("Error: Invalid PDF path provided to extract_financial_pages")
        return

    try:
        doc = pymupdf.open(pdf_path)
        new_doc = pymupdf.open()
    except Exception as e:
        print(f"Error opening PDF: {e}")
        return

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
    if not file_path:
        print("Error: Invalid file path for S3 upload")
        return

    try:
        s3_client.upload_file(file_path, bucket_name, os.path.basename(file_path))
        print(f'File {file_path} uploaded to {bucket_name}')
    except ClientError as e:
        print(f"Error uploading to S3: {e}")

def delete_file(pdf_file_path):
    if pdf_file_path and os.path.exists(pdf_file_path):
        os.remove(pdf_file_path)
        print(f"{pdf_file_path} has been deleted.")
    else:
        print(f"{pdf_file_path} does not exist.")

def delete_from_s3(pdf_file_path):
    try: 
        s3_client.head_object(Bucket=bucket_name, Key=pdf_file_path) 
        object_exists = True 
        print(f'The object {pdf_file_path} exists in bucket {bucket_name}.') 
    except ClientError as e: 
        if e.response['Error']['Code'] == '404': 
            object_exists = False 
            print(f'The object {pdf_file_path} does not exist in bucket {bucket_name}.') 
        else: 
            raise

    if object_exists: 
        s3_client.delete_object(Bucket=bucket_name, Key=pdf_file_path) 
        print(f'File {pdf_file_path} deleted from bucket {bucket_name}.')

init()

if __name__ == "__main__":
    pdf_to_json("test.pdf")
