import pandas as pd
import time
import json
import boto3
import yfinance as yf
import re
import random
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from financetoolkit import Toolkit  

client = boto3.client('bedrock-runtime', region_name='us-west-2')  

session = boto3.Session(
    aws_access_key_id='ASIA5HE2WWPHXBPFCFJL',
    aws_secret_access_key='7jb6JULemZHcS6T2gkQ/OrKU8n3NtNv0aeoAbpHm',
    aws_session_token='IQoJb3JpZ2luX2VjEGUaCXVzLWVhc3QtMSJHMEUCIAqxXLfvum0ODUZHXRIoBQ66hxx+0f7YzEB8NDO4JeEcAiEAuoZDHbA+mT7fB0k9kQ99NoJ6uEVL6egR1AnMBsgSe/sqogII3f//////////ARACGgw5MDg3MTAwMzIzMzUiDNNRlkYkif0A4jX1ICr2ATczjPJ87AlToRJyaR6KCVLNIlrZ2CL5KCOuqGDi9iQFwDtmLZEpOhXs3BHXHP9QhMwZNj0fuUajeMsuzWcG/WhSsOnaPe55+vHMZCmVpgwvtpT7ipsqgThif45xQuM36CV2eriB65wH3StDLby8uBKn2zWiL8ps4c2sNDlpmvOI4DyBnBputI140JYC8pZzYjwh1xfhLA4oGHjx/sEvFjawkLSwkWFC8iaX4KT/rX342mQVIHNaxmkeiH7XxoBHBjSnHWx9KsfwvHb7YwUHfjma9dhF8KVabzGYEg//qG+bu23oR16wC/sdv1Q3KNuvVihbxg/CQTDKsp+5BjqdAd4aj7bhKU3gRriBe+Wo9kWd33lH77575ykx8/im2bNAVvvijShg34TBqgPMevwyIGQIGdYdIU39cttybcQK3++gvouUzDsPONSoVweQVdp5GHVcrEL6n7Z+8anb3w9X52CdQG1+6eei5xIlBL9HyAL6dhs3GG/LDBiRMaLW40LJYzW6Sostg6RK0Ib+RjcAciBjvSpDkVioHcKghbE=',
    region_name='us-west-2'
)


client = session.client('bedrock-runtime')

# EXEMPLE TODO: LOAD DATA DU CSV EN DICT
csv_data = {
    'Report ID': ['Report_001'],
    'Company Name': ['PrivateTech'],
    'Fiscal Year': [2022],
    'Report Date': ['2022-12-31'],
    'Currency': ['USD'],
    'Summary': ['PrivateTech develops software solutions for e-commerce optimization in North America.'],
    'Total Revenue': [5006780],
    'Total Net Income': [707800],
    'Total Operating Income': [609800],
    'Total Expenses': [4309700],
    'Cost of Goods Sold (COGS)': [2500560],
    'Selling, General, and Administrative (SG&A)': [1225000],
    'Research and Development (R&D)': [400180],
    'Depreciation and Amortization': [200000],
    'Interest Expense': [100000],
    'Other Expenses': ['Not Present'],
    'Total Debt': [2000000],
    'Debt-to-Equity Ratio': [0.5],
    'Long-Term Debt': [1500000],
    'Short-Term Debt': [500000],
    'Total Equity': [4000000],
    'Gross Profit Margin': [0.55],
    'Operating Profit Margin': [0.12],
    'Net Profit Margin': [0.14],
    'Return on Assets (ROA)': [0.10],
    'Return on Equity (ROE)': [0.175]
}

company_data = pd.DataFrame(csv_data)


# Fetch industry benchmarks from Bedrock
def get_industry_benchmarks(company_summary):
    model_id = "anthropic.claude-v2"
    prompt_text = (
        f"Human: Provide industry averages as JSON with specific keys with the same format as the following format, replacing only the <value> fields and not adding any text, only return an answer in the format below without any modifications except for replacing the <value> fields:"
        f"{{'Industry Total Revenue': <value>, 'Industry Net Income': <value>, 'Industry Total Expenses': <value>, 'Industry Debt-to-Equity Ratio': <value>, 'Industry EBITDA': <value>}}.\n\n"
        f"Company Description: {company_summary}\n\nAssistant:"
    )
    payload = {"prompt": prompt_text, "max_tokens_to_sample": 50, "temperature": 0.5}
    expected_keys = ["Industry Total Revenue", "Industry Net Income", "Industry Total Expenses", "Industry Debt-to-Equity Ratio", "Industry EBITDA"]

    try:
        response = client.invoke_model(modelId=model_id, body=json.dumps(payload), contentType="application/json")
        response_body = response['body'].read().decode()
        benchmarks, missing_fields = validate_json_response(response_body, expected_keys)
        print("Fetched Industry Benchmarks:", benchmarks)  # Debug print statement
        return benchmarks, missing_fields
    except Exception as e:
        print(f"Error fetching industry benchmarks: {e}")
        return {key: "N/A (Data not available)" for key in expected_keys}, expected_keys





# Identify competitors using Amazon Bedrock
def get_competitors_from_bedrock(summary, retries=5, delay=1):
    model_id = "anthropic.claude-v2"
    prompt_text = (
        f"Human: Based on the following company description, identify a list of competitors and provide only their ticker symbols (e.g., AAPL, MSFT, GOOG).\n\n"
        f"Company Description: {summary}\n\nAssistant:"
    )
    payload = {
        "prompt": prompt_text,
        "max_tokens_to_sample": 150,
        "temperature": 0.7
    }

    for attempt in range(retries):
        try:
            response = client.invoke_model(
                modelId=model_id,
                body=json.dumps(payload),
                contentType="application/json"
            )
            response_body = json.loads(response['body'].read())
            competitors_text = response_body.get('completion', '').strip()

            # Use regex to extract possible ticker symbols from response
            competitors = re.findall(r'\b[A-Z]{1,5}\b', competitors_text)
            return competitors

        except client.exceptions.ThrottlingException:
            if attempt < retries - 1:
                # Wait with exponential backoff and jitter
                jitter = random.uniform(0, delay)  # Add a random delay to avoid synchronized retries
                wait_time = delay + jitter
                print(f"Throttling exception encountered. Retrying in {wait_time:.2f} seconds...")
                time.sleep(wait_time)
                delay *= 2  # Exponential backoff
            else:
                print("Maximum retries reached. Exiting.")
                raise
        except Exception as e:
            print(f"An error occurred: {e}")
            return None
        
def validate_json_response(response_str, expected_keys):
    try:
        response_json = json.loads(response_str)
        result = {key: response_json.get(key, "N/A") for key in expected_keys}
        # Missing fields should be only those not in response at all, not those intentionally set to "N/A"
        missing_fields = [key for key in expected_keys if key not in response_json or response_json[key] == ""]
        return result, missing_fields
    except json.JSONDecodeError:
        print("Error: Received response is not in the expected JSON format.")
    return {key: "N/A" for key in expected_keys}, expected_keys



# Function to validate ticker symbols
def validate_ticker(symbol):
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        # Check for the presence of 'shortName' as an indicator of a valid ticker
        if 'shortName' in info:
            return True
        else:
            return False
    except Exception as e:
        print(f"Error validating ticker {symbol}: {e}")
        return False


# Function to fetch financial data
def fetch_financial_data(symbol):
    try:
        stock = yf.Ticker(symbol)
        financials = stock.financials
        balance_sheet = stock.balance_sheet
        cash_flow = stock.cashflow

        # Extracting relevant financial metrics
        total_revenue = financials.loc['Total Revenue'].iloc[0] if 'Total Revenue' in financials.index else 'Not Available'
        net_income = financials.loc['Net Income'].iloc[0] if 'Net Income' in financials.index else 'Not Available'
        operating_income = financials.loc['Operating Income'].iloc[0] if 'Operating Income' in financials.index else 'Not Available'
        ebitda = financials.loc['EBITDA'].iloc[0] if 'EBITDA' in financials.index else 'Not Available'
        operating_cash_flow = cash_flow.loc['Total Cash From Operating Activities'].iloc[0] if 'Total Cash From Operating Activities' in cash_flow.index else 'Not Available'

        # Calculating Debt-to-Equity Ratio
        total_liabilities = balance_sheet.loc['Total Liab'].iloc[0] if 'Total Liab' in balance_sheet.index else None
        total_equity = balance_sheet.loc['Total Stockholder Equity'].iloc[0] if 'Total Stockholder Equity' in balance_sheet.index else None
        debt_to_equity_ratio = (total_liabilities / total_equity) if total_liabilities and total_equity else 'Not Available'

        return {
            'Symbol': symbol,
            'Total Revenue': total_revenue,
            'Net Income': net_income,
            'Operating Income': operating_income,
            'Debt-to-Equity Ratio': debt_to_equity_ratio,
            'EBITDA': ebitda,
            'Cash Flow': operating_cash_flow
        }
    except Exception as e:
        print(f"An error occurred fetching data for {symbol}: {e}")
        return {
            'Symbol': symbol,
            'Error': str(e)
        }


# Function to retrieve validated competitors and their data
def competitor_comparison(df):
    company_summary = df['Summary'].iloc[0]
    competitors = get_competitors_from_bedrock(company_summary)
    
    validated_competitors = [symbol for symbol in competitors if validate_ticker(symbol)]
    
    competitor_data = []
    for symbol in validated_competitors:
        competitor_info = fetch_financial_data(symbol)
        if competitor_info:
            # Remove 'Debt-to-Equity Ratio' and 'Cash Flow' fields if present
            competitor_info.pop("Debt-to-Equity Ratio", None)
            competitor_info.pop("Cash Flow", None)
            competitor_data.append(competitor_info)
        else:
            print(f"Warning: Financial data for {symbol} is unavailable or incomplete.")

    if not competitor_data:
        print("No valid competitor financial data was found.")
        return pd.DataFrame(columns=["Symbol", "Total Revenue", "Net Income", "Operating Income", "EBITDA"])

    competitor_df = pd.DataFrame(competitor_data)
    competitor_df.index = ["Competitor " + str(i + 1) for i in range(len(competitor_df))]
    
    return competitor_df



# Financial Health Summary
def financial_health_summary(df):
    required_fields = ["Total Revenue", "Total Net Income", "Total Expenses", "Debt-to-Equity Ratio", "Total Operating Income"]
    missing_fields = [field for field in required_fields if field not in df or pd.isna(df[field].iloc[0])]

    if missing_fields:
        return f"Financial Health Summary is not computable because {', '.join(missing_fields)} are missing."

    metrics = {
        "Total Revenue": df['Total Revenue'].iloc[0],
        "Total Net Income": df['Total Net Income'].iloc[0],
        "Total Expenses": df['Total Expenses'].iloc[0],
        "Debt-to-Equity Ratio": df['Debt-to-Equity Ratio'].iloc[0],
        "EBITDA": df['Total Operating Income'].iloc[0]
    }

    # Get industry benchmarks and check if it's a dictionary
    industry_benchmarks, missing_fields = get_industry_benchmarks(df['Summary'].iloc[0])
    if isinstance(industry_benchmarks, dict):
        metrics.update(industry_benchmarks)
    else:
        print("Error: Industry benchmarks format is incorrect.")
        missing_fields.append("Industry Benchmarks Format Error")

    return metrics



# Fetch industry expense ratios from Bedrock
def get_industry_expense_ratios_from_bedrock(company_summary):
    model_id = "anthropic.claude-v2"
    prompt_text = (
       f"Human: Provide industry averages for expenses as JSON with specific keys with the same format as the following format, replacing only the <value> fields and not adding any text, only return an answer in the format above without any modifications except for replacing the <value> fields:"
        f"{{'Industry COGS Ratio': <value>, 'Industry SG&A Ratio': <value>, 'Industry R&D Ratio': <value>, 'Industry Depreciation Ratio': <value>, 'Industry Interest Expense Ratio': <value>}}.\n\n"
        f"Company Description: {company_summary}\n\nAssistant:"
    )
    payload = {"prompt": prompt_text, "max_tokens_to_sample": 50, "temperature": 0.5}
    expected_keys = ["Industry COGS Ratio", "Industry SG&A Ratio", "Industry R&D Ratio", "Industry Depreciation Ratio", "Industry Interest Expense Ratio"]

    try:
        response = client.invoke_model(modelId=model_id, body=json.dumps(payload), contentType="application/json")
        response_body = response['body'].read().decode()
        return validate_json_response(response_body, expected_keys)
    except Exception as e:
        print(f"Error fetching industry expense ratios: {e}")
        return {key: "N/A (Data not available)" for key in expected_keys}, expected_keys


# Expense Breakdown
def expense_breakdown(df):
    required_fields = [
        "Cost of Goods Sold (COGS)", "Selling, General, and Administrative (SG&A)", 
        "Research and Development (R&D)", "Depreciation and Amortization", 
        "Interest Expense", "Other Expenses", "Total Expenses"
    ]
    missing_fields = [field for field in required_fields if field not in df or pd.isna(df[field].iloc[0])]

    if missing_fields:
        return f"Expense Breakdown is not computable because {', '.join(missing_fields)} are missing."

    expenses = {
        "COGS": df['Cost of Goods Sold (COGS)'].iloc[0],
        "SG&A": df['Selling, General, and Administrative (SG&A)'].iloc[0],
        "R&D": df['Research and Development (R&D)'].iloc[0],
        "Depreciation and Amortization": df['Depreciation and Amortization'].iloc[0],
        "Interest Expense": df['Interest Expense'].iloc[0],
        "Other Expenses": df['Other Expenses'].iloc[0]
    }
    total_expenses = df['Total Expenses'].iloc[0]
    expense_ratios = {k: (v / total_expenses * 100) if v != "Not Present" else "Not Present" for k, v in expenses.items()}

    industry_expense_ratios, _ = get_industry_expense_ratios_from_bedrock(df['Summary'].iloc[0])
    
    if isinstance(industry_expense_ratios, dict) and all(isinstance(v, str) for v in industry_expense_ratios.values()):
        expense_ratios.update(industry_expense_ratios)
    else:
        print("Error: Industry expense ratios format is incorrect.")
    return expense_ratios



# Valuation Estimation 
def estimate_company_valuation(df):
    required_fields = ["Total Revenue", "Total Net Income", "Total Expenses", "Debt-to-Equity Ratio", 
                       "Total Operating Income", "Total Equity"]
    missing_fields = [field for field in required_fields if field not in df or pd.isna(df[field].iloc[0])]

    if missing_fields:
        return f"Company Valuation is not computable because {', '.join(missing_fields)} are missing."

    # Retrieve all metrics to pass to Bedrock
    financial_metrics = {
        "Total Revenue": df['Total Revenue'].iloc[0],
        "Net Income": df['Total Net Income'].iloc[0],
        "Total Expenses": df['Total Expenses'].iloc[0],
        "Debt-to-Equity Ratio": df['Debt-to-Equity Ratio'].iloc[0],
        "EBITDA": df['Total Operating Income'].iloc[0]
    }
    company_summary = df['Summary'].iloc[0]
    
    # Request Bedrock for a dynamic revenue multiple based on comprehensive metrics
    revenue_multiple = get_revenue_multiple_from_bedrock(company_summary, financial_metrics)
    if revenue_multiple is None:
        print("Using default revenue multiple of 5.")
        revenue_multiple = 5  # Fallback to default if Bedrock fails
    
    revenue = df['Total Revenue'].iloc[0]
    valuation = revenue * revenue_multiple

    # Calculate Market-to-Book Ratio
    total_equity = df['Total Equity'].iloc[0]
    market_to_book_ratio = valuation / total_equity if total_equity != 0 else 'N/A'  # Avoid division by zero

    return valuation, market_to_book_ratio



# EPS Calculation 
def calculate_eps(df, estimated_shares):
    required_fields = ["Total Net Income"]
    missing_fields = [field for field in required_fields if field not in df or pd.isna(df[field].iloc[0])]

    if missing_fields:
        return f"EPS Calculation is not computable because {', '.join(missing_fields)} are missing."

    net_income = df['Total Net Income'].iloc[0]
    eps = net_income / estimated_shares
    return eps



# Enhanced Summary generation function with CONTEXT LINKING
def generate_summary_analysis(section_name, text, linked_metrics, retries=5, delay=1):
    model_id = "anthropic.claude-v2"
    
    prompt_text = (
        f"Human: Provide a detailed analysis and summary for the '{section_name}' section based on the following information.\n\n"
        f"{section_name} Details: {text}\n\n"
        f"In your response, link these insights with the following related metrics where relevant:\n\n{linked_metrics}\n\n"
        f"format you're answer in bulletpoints"
        f"your response should be of a maximum of 600 tokens and the last sentence should be complete"
        f"Assistant:"
    )

    payload = {
        "prompt": prompt_text,
        "max_tokens_to_sample": 600,
        "temperature": 0.7
    }

    # Retry mechanism to minimise retries
    for attempt in range(retries):
        try:
            response = client.invoke_model(
                modelId=model_id,
                body=json.dumps(payload),
                contentType="application/json"
            )
            response_body = json.loads(response['body'].read())
            analysis = response_body.get('completion', '').strip()
            return analysis

        except client.exceptions.ThrottlingException:
            if attempt < retries - 1:  # Only sleep if there are remaining attempts
                print(f"Throttling exception encountered. Retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= 2  # Exponential backoff
            else:
                print("Maximum retries reached. Exiting.")
                raise

        except Exception as e:
            print(f"An error occurred: {e}")
            return None
        
# Function to get revenue multiple based on company metrics from Bedrock
def get_revenue_multiple_from_bedrock(company_summary, financial_metrics, retries=5, delay=1):
    model_id = "anthropic.claude-v2"
    
    # Refined prompt to request only the numeric value for the multiple
    prompt_text = (
        f"Human: Based on the following company profile and financial metrics, suggest a reasonable revenue multiple for valuation purposes. "
        f"Only provide the numeric value of the revenue multiple in your answer.\n\n"
        f"Company Description: {company_summary}\n\n"
        f"Financial Metrics: {financial_metrics}\n\n"
        f"Assistant:"
    )
    
    payload = {
        "prompt": prompt_text,
        "max_tokens_to_sample": 10,  # Reduced token sample since we expect only a number
        "temperature": 0.7
    }

    for attempt in range(retries):
        try:
            response = client.invoke_model(
                modelId=model_id,
                body=json.dumps(payload),
                contentType="application/json"
            )
            response_body = json.loads(response['body'].read())
            multiple_text = response_body.get('completion', '').strip()

            # Directly return the numeric multiple as a float if the response contains it
            return float(multiple_text)

        except ValueError:
            print("The response did not contain a valid number for the revenue multiple.")
            return None
        except client.exceptions.ThrottlingException:
            if attempt < retries - 1:
                jitter = random.uniform(0, delay)
                wait_time = delay + jitter
                print(f"Throttling exception encountered. Retrying in {wait_time:.2f} seconds...")
                time.sleep(wait_time)
                delay *= 2  # Exponential backoff
            else:
                print("Maximum retries reached. Exiting.")
                raise
        except Exception as e:
            print(f"An error occurred: {e}")
            return None


# Modified Valuation Estimation function using Bedrock response for revenue multiple
def estimate_company_valuation(df):
    # Retrieve all metrics to pass to Bedrock
    financial_metrics = {
        "Total Revenue": df['Total Revenue'].iloc[0],
        "Net Income": df['Total Net Income'].iloc[0],
        "Total Expenses": df['Total Expenses'].iloc[0],
        "Debt-to-Equity Ratio": df['Debt-to-Equity Ratio'].iloc[0],
        "EBITDA": df['Total Operating Income'].iloc[0]
    }
    company_summary = df['Summary'].iloc[0]
    
    # Request Bedrock for a dynamic revenue multiple based on comprehensive metrics
    revenue_multiple = get_revenue_multiple_from_bedrock(company_summary, financial_metrics)
    if revenue_multiple is None:
        print("Using default revenue multiple of 5.")
        revenue_multiple = 5  # Fallback to default if Bedrock fails
    
    revenue = df['Total Revenue'].iloc[0]
    valuation = revenue * revenue_multiple

    # Calculate Market-to-Book Ratio
    total_equity = df['Total Equity'].iloc[0]
    market_to_book_ratio = valuation / total_equity if total_equity != 0 else 'N/A'  # Avoid division by zero

    # Return both the valuation and market-to-book ratio as a tuple
    return valuation, market_to_book_ratio

def mean_absolute_error(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate the absolute errors and return the mean
    return np.mean(np.abs(y_true - y_pred))

def stock_price_prediction(estimated_shares=1000000):
    # Estimate valuation and market-to-book ratio using dynamic revenue multiple
    company_valuation, market_to_book_ratio = estimate_company_valuation(company_data)

    # Use only the valuation (first element of the tuple) for IPO stock price calculation
    valuation = company_valuation

    # Updated data with valuation as target for prediction
    target_ipo_prices = [valuation]

    # Organize data into features for prediction
    data = {
        'Total_Revenue': [company_data['Total Revenue'].iloc[0]],
        'Net_Income': [company_data['Total Net Income'].iloc[0]],
        'Total_Expenses': [company_data['Total Expenses'].iloc[0]],
        'Debt_to_Equity_Ratio': [company_data['Debt-to-Equity Ratio'].iloc[0]],
        'EBITDA': [company_data['Total Operating Income'].iloc[0]]
    }
    df = pd.DataFrame(data)

    # Check if only one sample is available
    if len(df) > 1:
        X_train, X_test, y_train, y_test = train_test_split(df, target_ipo_prices, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
    else:
        model = LinearRegression()
        model.fit(df, target_ipo_prices)
        predictions = model.predict(df)
        mae = 0  # MAE is not meaningful with a single data point

    # Calculate IPO stock price from the valuation
    ipo_stock_price = valuation / estimated_shares
    return mae, predictions, valuation, ipo_stock_price

def calculate_valuation_ratios(df):
    required_fields = ["Total Revenue", "Total Net Income", "Total Equity"]
    missing_fields = [field for field in required_fields if field not in df or pd.isna(df[field].iloc[0])]

    if missing_fields:
        return f"Valuation Ratios are not computable because {', '.join(missing_fields)} are missing."

    valuation_ratios = {}
    try:
        total_revenue = df['Total Revenue'].iloc[0]
        net_income = df['Total Net Income'].iloc[0]
        total_equity = df['Total Equity'].iloc[0]

        # Price-to-Earnings Ratio (P/E) - Assuming a placeholder price or IPO valuation
        valuation_ratios['Price-to-Earnings Ratio (P/E)'] = total_revenue / net_income if net_income else 'N/A'
        
        # Price-to-Sales Ratio (P/S) - using valuation
        valuation_ratios['Price-to-Sales Ratio (P/S)'] = total_revenue / total_revenue if total_revenue else 'N/A'

        # Price-to-Book Ratio (P/B)
        valuation_ratios['Price-to-Book Ratio (P/B)'] = total_revenue / total_equity if total_equity else 'N/A'
        
    except KeyError as e:
        print(f"Missing data for valuation ratio calculation: {e}")
    
    return valuation_ratios

# Risk Metrics Calculation (Custom Function)
def calculate_risk_metrics_with_thresholds(df):
    risk_metrics = {
        "Debt-to-Equity Ratio": df['Debt-to-Equity Ratio'].iloc[0],
        "Profit Margin": df['Net Profit Margin'].iloc[0]
    }
    
    thresholds = {
        "Debt-to-Equity Ratio": {"Low": 0.5, "High": 1.5},
        "Profit Margin": {"Low": 0.1, "High": 0.3}
    }
    
    # Create a new dictionary to store risk metrics along with their categories
    categorized_risk_metrics = risk_metrics.copy()
    
    for metric, value in risk_metrics.items():
        if metric in thresholds:
            if value < thresholds[metric]["Low"]:
                categorized_risk_metrics[f"{metric} Risk Category"] = "Low"
            elif value > thresholds[metric]["High"]:
                categorized_risk_metrics[f"{metric} Risk Category"] = "High"
            else:
                categorized_risk_metrics[f"{metric} Risk Category"] = "Moderate"
    
    return categorized_risk_metrics




# Forecasting Placeholder (Simple Growth Estimate Function)
def forecast_growth(df, growth_rate=0.05):
    required_fields = ["Total Revenue", "Total Net Income"]
    missing_fields = [field for field in required_fields if field not in df or pd.isna(df[field].iloc[0])]

    if missing_fields:
        return f"Growth Forecast is not computable because {', '.join(missing_fields)} are missing."

    forecasts = {}
    try:
        total_revenue = df['Total Revenue'].iloc[0]
        net_income = df['Total Net Income'].iloc[0]

        # Simple growth projections based on a static growth rate
        forecasts['Forecasted Revenue Growth'] = total_revenue * (1 + growth_rate)
        forecasts['Forecasted Net Income Growth'] = net_income * (1 + growth_rate)

    except KeyError as e:
        print(f"Missing data for forecast calculation: {e}")
    
    return forecasts

# Fetch industry revenue from Bedrock
def get_industry_revenue_from_bedrock(company_summary):
    model_id = "anthropic.claude-v2"
    prompt_text = (
        f"Human: Provide the total annual revenue of the industry as a JSON with the following format: "
        f"{{'Industry Revenue': <value>}}.\n\nCompany Description: {company_summary}\n\nAssistant:"
    )
    payload = {"prompt": prompt_text, "max_tokens_to_sample": 20, "temperature": 0.5}
    expected_keys = ["Industry Revenue"]

    try:
        response = client.invoke_model(modelId=model_id, body=json.dumps(payload), contentType="application/json")
        response_body = response['body'].read().decode()
        industry_data, missing_fields = validate_json_response(response_body, expected_keys)
        return industry_data.get('Industry Revenue'), missing_fields
    except Exception as e:
        print(f"Error fetching industry revenue: {e}")
        return "N/A (Data not available)", expected_keys

# Fetch industry growth from Bedrock
def get_industry_growth_from_bedrock(company_summary):
    model_id = "anthropic.claude-v2"
    prompt_text = (
        f"Human: Provide the year-over-year growth rate for the industry as JSON with this format: "
        f"{{'Industry Growth Rate': <value>}}.\n\nCompany Description: {company_summary}\n\nAssistant:"
    )
    payload = {"prompt": prompt_text, "max_tokens_to_sample": 20, "temperature": 0.5}
    expected_keys = ["Industry Growth Rate"]

    try:
        response = client.invoke_model(modelId=model_id, body=json.dumps(payload), contentType="application/json")
        response_body = response['body'].read().decode()
        industry_data, missing_fields = validate_json_response(response_body, expected_keys)
        return industry_data.get('Industry Growth Rate'), missing_fields
    except Exception as e:
        print(f"Error fetching industry growth rate: {e}")
        return "N/A (Data not available)", expected_keys


# Industry Valuation Ratios (P/E, P/S) using Bedrock
def get_industry_valuation_ratios_from_bedrock(company_summary, retries=5, delay=1):
    model_id = "anthropic.claude-v2"
    prompt_text = (
        f"Human: Based on the company description below, provide the average Price-to-Earnings (P/E) and Price-to-Sales (P/S) ratios for the industry.\n\n"
        f"Company Description: {company_summary}\n\nAssistant:"
    )
    payload = {
        "prompt": prompt_text,
        "max_tokens_to_sample": 50,
        "temperature": 0.7
    }
    for attempt in range(retries):
        try:
            response = client.invoke_model(modelId=model_id, body=json.dumps(payload), contentType="application/json")
            response_body = json.loads(response['body'].read())
            ratios_text = response_body.get('completion', '').strip()
            pe_match = re.search(r'P/E.*?(\d+(\.\d+)?)', ratios_text)
            ps_match = re.search(r'P/S.*?(\d+(\.\d+)?)', ratios_text)
            return {
                "Industry P/E": float(pe_match.group(1)) if pe_match else 'N/A',
                "Industry P/S": float(ps_match.group(1)) if ps_match else 'N/A'
            }
        except Exception as e:
            print(f"Error fetching industry valuation ratios: {e}")
            if attempt < retries - 1:
                time.sleep(delay)
                delay *= 2
    return None

# Fetch industry beta from Bedrock
def get_beta_from_bedrock(company_summary):
    model_id = "anthropic.claude-v2"
    prompt_text = (
        f"Human: Provide the industry beta (volatility measure) as JSON in this format: "
        f"{{'Industry Beta': <value>}}.\n\nCompany Description: {company_summary}\n\nAssistant:"
    )
    payload = {"prompt": prompt_text, "max_tokens_to_sample": 20, "temperature": 0.5}
    expected_keys = ["Industry Beta"]

    try:
        response = client.invoke_model(modelId=model_id, body=json.dumps(payload), contentType="application/json")
        response_body = response['body'].read().decode()
        industry_data, missing_fields = validate_json_response(response_body, expected_keys)
        return industry_data.get('Industry Beta'), missing_fields
    except Exception as e:
        print(f"Error fetching industry beta: {e}")
        return "N/A (Data not available)", expected_keys


# Emerging Market Trends using Bedrock
def get_emerging_market_trends_from_bedrock(company_summary, retries=5, delay=1):
    model_id = "anthropic.claude-v2"
    prompt_text = (
        f"Human: Based on the company description below, list emerging market trends relevant to this company.\n\n"
        f"Company Description: {company_summary}\n\nAssistant:"
    )
    payload = {
        "prompt": prompt_text,
        "max_tokens_to_sample": 100,
        "temperature": 0.7
    }
    for attempt in range(retries):
        try:
            response = client.invoke_model(modelId=model_id, body=json.dumps(payload), contentType="application/json")
            response_body = json.loads(response['body'].read())
            trends_text = response_body.get('completion', '').strip()
            return trends_text.split(', ')  # Assuming Bedrock returns a comma-separated list
        except Exception as e:
            print(f"Error fetching emerging market trends: {e}")
            if attempt < retries - 1:
                time.sleep(delay)
                delay *= 2
    return None

# Industry Revenue with Explicit Market Mention
def get_industry_revenue_from_bedrock(company_summary, retries=5, delay=1):
    model_id = "anthropic.claude-v2"
    prompt_text = (
        f"Human: Based on the company description below, identify the relevant industry or market sector and provide only the total annual revenue for this industry in USD as a numeric value. "
        f"Also, indicate the market sector (e.g., 'e-commerce' or 'software solutions') in your response. Format your answer as follows and don't add or remove anything else:\n\n"
        f"Industry: <market sector>, Revenue: <numeric value>\n\n"
        f"Company Description: {company_summary}\n\nAssistant:"
    )
    payload = {
        "prompt": prompt_text,
        "max_tokens_to_sample": 50,
        "temperature": 0.3
    }
    for attempt in range(retries):
        try:
            response = client.invoke_model(modelId=model_id, body=json.dumps(payload), contentType="application/json")
            response_body = json.loads(response['body'].read())
            industry_revenue_text = response_body.get('completion', '').strip()
            
            # Example format check: "Industry: E-commerce, Revenue: 10000000"
            match = re.match(r"Industry: (.+), Revenue: (\d+(\.\d+)?)", industry_revenue_text)
            if match:
                industry = match.group(1).strip()
                revenue = float(match.group(2))
                return industry, revenue
            print("Received invalid format for industry revenue.")
            return None, None
        except Exception as e:
            print(f"Error fetching industry revenue: {e}")
            if attempt < retries - 1:
                time.sleep(delay)
                delay *= 2
    return None, None

    
# Beta Value using Bedrock
def get_beta_from_bedrock(company_summary, retries=5, delay=1):
    model_id = "anthropic.claude-v2"
    prompt_text = (
        f"Human: Based on the company description below, provide only the beta (volatility measure) for the industry as a numeric value (e.g., '1.2'). "
        f"Do not include any additional text.\n\n"
        f"Company Description: {company_summary}\n\nAssistant:"
    )
    payload = {
        "prompt": prompt_text,
        "max_tokens_to_sample": 20,
        "temperature": 0.3
    }
    for attempt in range(retries):
        try:
            response = client.invoke_model(modelId=model_id, body=json.dumps(payload), contentType="application/json")
            response_body = json.loads(response['body'].read())
            beta_text = response_body.get('completion', '').strip()
            
            # Check if the response is a valid number
            if beta_text.replace('.', '', 1).isdigit():  # Allows one decimal point
                return float(beta_text), []  # No missing fields
            else:
                print("Received invalid number format for beta value.")
                return None, ["Beta Value"]  # Indicate missing beta value
            
        except Exception as e:
            print(f"Error fetching beta value: {e}")
            if attempt < retries - 1:
                time.sleep(delay)
                delay *= 2
    return None, ["Beta Value"]

# Industry Growth Rate with Explicit Market Mention
def get_industry_growth_from_bedrock(company_summary, retries=5, delay=1):
    model_id = "anthropic.claude-v2"
    prompt_text = (
        f"Human: Based on the company description below, identify the relevant industry or market sector and provide only the Year-over-Year revenue growth rate as a percentage. "
        f"Format your answer as follows and don't add or remove anything else:\n\n"
        f"Industry: <market sector>, Growth Rate: <numeric value>\n\n"
        f"Company Description: {company_summary}\n\nAssistant:"
    )
    payload = {
        "prompt": prompt_text,
        "max_tokens_to_sample": 50,
        "temperature": 0.3
    }
    for attempt in range(retries):
        try:
            response = client.invoke_model(modelId=model_id, body=json.dumps(payload), contentType="application/json")
            response_body = json.loads(response['body'].read())
            growth_text = response_body.get('completion', '').strip()
            
            # Example format check: "Industry: E-commerce, Growth Rate: 5"
            match = re.match(r"Industry: (.+), Growth Rate: (\d+(\.\d+)?)", growth_text)
            if match:
                industry = match.group(1).strip()
                growth_rate = float(match.group(2))
                return industry, growth_rate
            print("Received invalid format for industry growth rate.")
            return None, None
        except Exception as e:
            print(f"Error fetching industry growth rate: {e}")
            if attempt < retries - 1:
                time.sleep(delay)
                delay *= 2
    return None, None





def get_industry_valuation_ratios_from_bedrock(company_summary, retries=5, delay=1):
    model_id = "anthropic.claude-v2"
    prompt_text = (
        f"Human: Based on the company description below, provide the industry Price-to-Earnings (P/E) ratio and Price-to-Sales (P/S) ratio as numeric values. "
        f"Return in the following JSON format only: {{'Industry P/E': <P/E value>, 'Industry P/S': <P/S value>}}.\n\n"
        f"Company Description: {company_summary}\n\nAssistant:"
    )
    payload = {
        "prompt": prompt_text,
        "max_tokens_to_sample": 50,
        "temperature": 0.5
    }
    for attempt in range(retries):
        try:
            response = client.invoke_model(modelId=model_id, body=json.dumps(payload), contentType="application/json")
            response_body = json.loads(response['body'].read())
            ratios_text = response_body.get('completion', '').strip()
            
            # Validate JSON format
            try:
                industry_ratios = json.loads(ratios_text)
                if all(k in industry_ratios for k in ["Industry P/E", "Industry P/S"]):
                    return industry_ratios
            except json.JSONDecodeError:
                print("Received invalid JSON format for industry valuation ratios.")
            return {"Industry P/E": "N/A", "Industry P/S": "N/A"}

        except Exception as e:
            print(f"Error fetching industry valuation ratios: {e}")
            if attempt < retries - 1:
                time.sleep(delay)
                delay *= 2
    return {"Industry P/E": "N/A", "Industry P/S": "N/A"}


# Integrate the improved Bedrock function calls in the analysis
# def display_analysis(df):
#     linked_metrics = {}
    
#     health_summary = financial_health_summary(df)
#     expenses = expense_breakdown(df)
#     competitor_df = competitor_comparison(company_data)
#     valuation, market_to_book_ratio = estimate_company_valuation(df)  
#     estimated_shares = 1000000  # EXAMPLE: can be customized as needed
#     eps = calculate_eps(df, estimated_shares)
#     mae, predictions, company_valuation, ipo_stock_price = stock_price_prediction(estimated_shares=estimated_shares)
    
#     print("=== Analysis Report ===\n")
    
#     print("1. Financial Health Summary:")
#     print("   --------------------------")
#     for key, value in health_summary.items():
#         print(f"   {key}: {value}")
#     linked_metrics['Financial Health Summary'] = health_summary
#     print("   Detailed Analysis:")
#     print("   " + generate_summary_analysis("Financial Health Summary", health_summary, linked_metrics).replace("\n", "\n   "))
    
#     time.sleep(1)
    
#     print("\n2. Expense Breakdown:")
#     print("   -------------------")
#     for key, value in expenses.items():
#         print(f"   {key}: {value}%")
#     linked_metrics['Expense Breakdown'] = expenses
#     print("   Detailed Analysis:")
#     print("   " + generate_summary_analysis("Expense Breakdown", expenses, linked_metrics).replace("\n", "\n   "))
    
#     time.sleep(1)

#     print("\n3. Competitor Comparison:")
#     print("   -----------------------")
#     print("   Competitors Data:")
#     print("   " + competitor_df.to_string(index=False).replace("\n", "\n   "))
#     linked_metrics['Competitor Comparison'] = competitor_df.to_dict()
#     print("   Detailed Analysis:")
#     print("   " + generate_summary_analysis("Competitor Comparison", competitor_df.to_string(index=False), linked_metrics).replace("\n", "\n   "))
    
#     time.sleep(1)

#     print("\n4. Estimated Company Valuation:")
#     print("   ----------------------------")
#     print(f"   Estimated Valuation: ${valuation:,.2f}")
#     print(f"   Market-to-Book Ratio: {market_to_book_ratio:.2f}" if market_to_book_ratio != 'N/A' else "   Market-to-Book Ratio: N/A")
#     linked_metrics['Company Valuation'] = {"Estimated Valuation": f"${valuation:,.2f}"}
#     print("   Detailed Analysis:")
#     print("   " + generate_summary_analysis("Company Valuation", f"Estimated Valuation: ${valuation:,.2f}", linked_metrics).replace("\n", "\n   "))
    
#     time.sleep(1)

#     print("\n5. Earnings Per Share (EPS):")
#     print("   --------------------------")
#     print(f"   EPS: ${eps:.2f}")
#     linked_metrics['Earnings Per Share'] = {"EPS": f"${eps:.2f}"}
#     print("   Detailed Analysis:")
#     print("   " + generate_summary_analysis("Earnings Per Share", f"Earnings Per Share (EPS): ${eps:.2f}", linked_metrics).replace("\n", "\n   "))
    
#     time.sleep(1)

#     print("\n6. Market Analysis:")
#     print("   ------------------")
    
#     # Market Share Calculation using Bedrock
#     industry, industry_revenue = get_industry_revenue_from_bedrock(df['Summary'].iloc[0])
#     if industry_revenue is not None:
#         market_share = calculate_market_share(df['Total Revenue'].iloc[0], industry_revenue)
#         print(f"   Industry: {industry}")
#         print(f"   Market Share: {market_share}%")
#     else:
#         print("   Market Share: N/A%")

#     # Industry Revenue Growth
#     industry, industry_growth_rate = get_industry_growth_from_bedrock(df['Summary'].iloc[0])
#     if industry_growth_rate is not None:
#         print(f"   Industry: {industry}")
#         print(f"   Industry Revenue Growth (YoY): {industry_growth_rate}%")
#     else:
#         print("   Industry Revenue Growth (YoY): N/A")

#     # Industry Valuation Ratios
#     industry_valuation_ratios = get_industry_valuation_ratios_from_bedrock(df['Summary'].iloc[0])
#     print("   Industry Valuation Ratios:")
#     for key, value in industry_valuation_ratios.items():
#         print(f"      {key}: {value}")
    
#     # Beta Value for Risk Analysis
#     beta_value = get_beta_from_bedrock(df['Summary'].iloc[0])
#     print(f"   Beta (Volatility): {beta_value}" if beta_value else "   Beta (Volatility): N/A")
    
#     # Emerging Market Trends
#     emerging_markets = get_emerging_market_trends_from_bedrock(df['Summary'].iloc[0])
#     print("   Emerging Market Trends:")
#     for trend in emerging_markets:
#         print(f"      - {trend}")
    
#     print("\n7. Estimated Company Valuation Summary:")
#     print("   -------------------------------------")
#     print(f"   Final Estimated Valuation: ${company_valuation:,.2f}")
#     print(f"   Mean Absolute Error: {mae:.2f}")
#     print(f"   Predicted IPO Valuation: ${predictions[0]:,.2f}")
#     print(f"   Estimated IPO Stock Price per Share: ${ipo_stock_price:.2f}")
    
#     print("\n=== End of Analysis Report ===\n")

# # Call display_analysis function to print analyses
# display_analysis(company_data)

# Function to calculate market share
def calculate_market_share(company_revenue, industry_revenue):
    try:
        if isinstance(industry_revenue, (int, float)) and industry_revenue > 0:
            return round((company_revenue / industry_revenue) * 100, 2)
    except TypeError:
        print("Error: industry_revenue must be a numeric value.")
    return "N/A"


# JSON generation function with serialization handling
def convert_to_serializable(obj):
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    elif isinstance(obj, pd.Series):
        return obj.to_dict()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
    elif isinstance(obj, np.int64) or isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.float64) or isinstance(obj, np.floating):
        return float(obj)
    return obj

# Unpacking and organizing output in generate_json_output
def generate_json_output(df):
    industry_benchmarks, health_missing_fields = get_industry_benchmarks(df['Summary'].iloc[0])
    industry_expenses, expense_missing_fields = get_industry_expense_ratios_from_bedrock(df['Summary'].iloc[0])
    
    industry_revenue, revenue_missing_fields = get_industry_revenue_from_bedrock(df['Summary'].iloc[0])
    if industry_revenue is None:
        industry_revenue = "N/A"
        revenue_missing_fields = ["Industry Revenue"]

    industry_growth, growth_missing_fields = get_industry_growth_from_bedrock(df['Summary'].iloc[0])
    if industry_growth is None:
        industry_growth = "N/A"
        growth_missing_fields = ["Industry Growth Rate"]

    beta_value, beta_missing_fields = get_beta_from_bedrock(df['Summary'].iloc[0])

    health_summary = {k: v for k, v in financial_health_summary(df).items() if v != "N/A"}
    expenses = {k: v for k, v in expense_breakdown(df).items() if v != "N/A"}
    competitor_df = competitor_comparison(df)
    valuation, market_to_book_ratio = estimate_company_valuation(df)
    eps = calculate_eps(df, estimated_shares=1000000)
    mae, predictions, company_valuation, ipo_stock_price = stock_price_prediction(estimated_shares=1000000)

    output = {
        "Financial Health Summary": {
            "Metrics": health_summary,
            "Industry Benchmarks": {k: v for k, v in industry_benchmarks.items() if v != "N/A"},
            "Missing Fields": health_missing_fields,
            "Detailed Analysis": generate_summary_analysis("Financial Health Summary", health_summary, {})
        },
        "Expense Breakdown": {
            "Expenses": expenses,
            "Industry Expense Ratios": {k: v for k, v in industry_expenses.items() if v != "N/A"},
            "Missing Fields": expense_missing_fields,
            "Detailed Analysis": generate_summary_analysis("Expense Breakdown", expenses, {})
        },
        "Competitor Comparison": {
            "Competitors Data": convert_to_serializable(competitor_df),
            "Detailed Analysis": generate_summary_analysis("Competitor Comparison", competitor_df.to_string(index=False), {})
        },
        "Company Valuation": {
            "Estimated Valuation": f"${valuation:,.2f}",
            "Market-to-Book Ratio": f"{market_to_book_ratio:.2f}" if market_to_book_ratio != 'N/A' else "N/A",
            "Detailed Analysis": generate_summary_analysis("Company Valuation", f"Estimated Valuation: ${valuation:,.2f}", {})
        },
        "Earnings Per Share (EPS)": {
            "EPS": f"${eps:.2f}",
            "Detailed Analysis": generate_summary_analysis("Earnings Per Share", f"Earnings Per Share (EPS): ${eps:.2f}", {})
        },
        "Market Analysis": {
            "Market Share": calculate_market_share(df['Total Revenue'].iloc[0], industry_revenue),
            "Industry Revenue Growth": industry_growth,
            "Beta (Volatility)": beta_value,
            "Missing Fields": {
                "Industry Revenue": revenue_missing_fields if isinstance(revenue_missing_fields, list) else ["Industry Revenue"],
                "Industry Growth Rate": growth_missing_fields if isinstance(growth_missing_fields, list) else ["Industry Growth Rate"],
                "Industry Beta": beta_missing_fields
            }
        },
        "Estimated Company Valuation Summary": {
            "Final Estimated Valuation": f"${company_valuation:,.2f}",
            "Mean Absolute Error": f"{mae:.2f}",
            "Predicted IPO Valuation": f"${predictions[0]:,.2f}" if predictions.size > 0 else "N/A",
            "Estimated IPO Stock Price per Share": f"${ipo_stock_price:.2f}"
        }
    }

    return json.dumps(convert_to_serializable(output), indent=4)

# Call generate_json_output and print
json_output = generate_json_output(company_data)
print("=== JSON Output for UI ===")
print(json_output)