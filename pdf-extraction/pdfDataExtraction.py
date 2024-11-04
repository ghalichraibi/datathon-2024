import boto3
import json
import fitz
from marker.convert import convert_single_pdf
from marker.models import load_all_models

session = boto3.Session(
    aws_access_key_id='ASIA5HE2WWPH2RKKMPHN',
    aws_secret_access_key='dMc1g3JKgUVejh94gq8+fcgoIPmqGNdGXLc4YHiq',
    aws_session_token='IQoJb3JpZ2luX2VjEFAaCXVzLWVhc3QtMSJGMEQCICkNKshbOLvWeIvMDia3GbgimKt7weM2xeipFa4t25apAiBTB+zqrf0yCEPM1f+eQk4xWW4RW41DJH4fWNPKi5O/LCqiAgjJ//////////8BEAIaDDkwODcxMDAzMjMzNSIMUuxIwg4zbmrqjRmhKvYB2j8N8Gub+S3TrgfvpsA/guoSOoIj/s0BlH6GMTGRS1nSqz+4gATg7VRAn0Rrn4Lb6h5Mr/uXni4Bwu7o49VtNQauNYLeL/mHaoPCrWw1ocAuPynryarlk+Dtg4DEqHHaJDDQvG1+KKIQa4UnOdF83XKHzHqD2rtLwtrntLZwhnBO1QFj3WUBolRmrAQa2UxYHbQkT/l5UYnNKKccDBWZgZhdjV3fBvTXMn7ZsLHyIQJIjB2QKW3ANgu9gj9Pj4aXT9rVKgCdBEY5thTqWkyXBGqCytu9mpqLuGrmWzss/6QnVTpQOSqGLqShAPTDb7Q5WUPVJ8ocMKDsmrkGOp4BrSTVvzbPF/KVglrmhh9V9vsKfXi1fZtk+J4rEpGGigjxWQSgDBX3FnHNLAJLYlgd6+XlsE96PX7CXSOnj3Xrc26Ap8kg0Rir+mWyBhnKJ55Apsx77/BCs5pWOYyn5pw8TbXnFD4Ep0SWWPoGQklMElPkFy6SbjOMJG+o47vbHP2XSVWyGQTS4bw6r4mRWxBRX9LP+ZHBA91SBmmEVSg=',  # Only if using temporary credentials
    region_name='us-west-2'  # Replace with your desired region
)

client = session.client('bedrock-runtime')

json_template = """
 {
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
}

"""

client = session.client('bedrock-runtime')


def get_competitors_from_bedrock(summary):
    model_id = "anthropic.claude-v2"  # Replace with the appropriate model ID

    # Construct the prompt with "Human:" prefix as required by Claude
    prompt_text = (
        f"Human: Fill the following json template with values found in the company financial report. Your response should not contain anything else than the filled out template. If you cannot find a value, put a null value in the appropriate field. For the summary field, write a quick summary of the company's activity sector based on its name\n\n"
        f"Company financial report: {summary}\n\n"
        f"json template: ${json_template}\n\nAssistant:"
    )

    # Define the request payload
    payload = {
        "prompt": prompt_text,
        "max_tokens_to_sample": 300,  # Adjust based on expected response length
        "temperature": 0.7  # Adjust to control response creativity
    }

    # Invoke the model
    response = client.invoke_model(
        modelId=model_id,
        body=json.dumps(payload),
        contentType="application/json"
    )

    # Parse the response
    response_body = json.loads(response['body'].read())
    competitors_text = response_body.get('completion', '').strip()

    # Process the response to extract competitors
    competitors = [comp.strip() for comp in competitors_text.split(',') if comp.strip()]

    return competitors

def pdf_to_string(pdf_path):
    text = ""
    with fitz.open(pdf_path) as pdf:
        for page_num in range(pdf.page_count):
            page = pdf[page_num]
            text += page.get_text()
    return text


if __name__ == '__main__':
    model_lst = load_all_models()
    md_text, images, out_meta = convert_single_pdf('./financial_data_scraped.pdf',model_lst)
    print(get_competitors_from_bedrock(md_text))