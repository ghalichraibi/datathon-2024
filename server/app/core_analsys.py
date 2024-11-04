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

client = boto3.client('bedrock-runtime', region_name='us-west-2')  

session = boto3.Session(
    aws_access_key_id='ASIA5HE2WWPHQPL5LEJL',
    aws_secret_access_key='3XRe1A1fyftCHAQTppDoHDIZoHe28wGYFApZbmUc',
    aws_session_token='IQoJb3JpZ2luX2VjEHEaCXVzLWVhc3QtMSJIMEYCIQC2YEQ8a6GD3tXYtYSeVLYL2jVw+EVNTHkM+LOaXKCHjQIhALFOViDbPl5TlkzZ2fO2Vqx/aUGGDr0Rtwa1QK/BclXLKqICCOr//////////wEQAhoMOTA4NzEwMDMyMzM1Igyel8ipyRVbMe2VgZsq9gHNvOXzr8EBBdU+k5Bg8LjMFuMuKSYiT76z7hGgBfA78QAT6HCOFIcj6oL2ANOpAObKYwAjSKcqz/FWntSq5FzHvS1vpJER7efQgF+VqN2XdSAxND/JCmvhUoRR57L0D0Hb1VurhFk3W5p88QW6NTOfw6z6/GewCta9fddjUaRtv9wJU/GNS1GbnfZ2TGNlD7EI+fFCC6G42lghJ/L4VcN0lBD/kAsEtLCltcCzUSKDMsjnXeGemxGMDm+M9XCY3WHlR51cP0DuvGfL8qxuX2pkBAvZ3ZDyQXSdav1bhigCv71q4Y2qjt/aHVP4RCCNurdYHzBtMbgw6o6iuQY6nAHoVd5+Sh8v3BZcyXcKlxekzA7SCf3FAqvYGOWugYjmKtPHuMv8XCitts9devWB74C9xcysjiSewkCA7LjSORgir6NfNksuukQjt+CTKZVAOwN6/gFspN34D1C84HOj1bx2JUlbhJecMkfn4RY9tPKNJ4AeO2T2tbKRvXHoJe720kuRQarHC7iXO2CpVP91Dtb3dep9U1w8O4+e65U=',
    region_name='us-west-2'
)

client = session.client('bedrock-runtime')

# Enhanced error handling function
def safe_access(df, column_name):
    """Safely access a DataFrame column and return its value or a placeholder."""
    try:
        if column_name in df.columns:
            return df[column_name].iloc[0]
        else:
            return "Not Present"
    except Exception as e:
        print(f"Error accessing column '{column_name}': {e}")
        return "Not Present"

# Fetch industry benchmarks from Bedrock
def get_industry_benchmarks(company_summary):
    model_id = "anthropic.claude-v2"
    prompt_text = (
        f"Human: Provide industry averages as JSON with specific keys with the same format as the following format, replacing only the <value> fields and not adding any text, only return an answer in the format below without any modifications except for replacing the <value> fields:"
        f"{{'Industry Total Revenue': <value>, 'Industry Net Income': <value>, 'Industry Total Expenses': <value>, 'Industry Debt-to-Equity Ratio': <value>, 'Industry EBITDA': <value>}}.\n\n"
        f"Company Description: {company_summary}\n\nAssistant:"
    )
    payload = {"prompt": prompt_text, "max_tokens_to_sample": 1000, "temperature": 0.5}
    expected_keys = ["Industry Total Revenue", "Industry Net Income", "Industry Total Expenses", "Industry Debt-to-Equity Ratio", "Industry EBITDA"]

    try:
        response = client.invoke_model(modelId=model_id, body=json.dumps(payload), contentType="application/json")
        response_body = response['body'].read().decode()

        completion_text = json.loads(response_body).get('completion', '').strip()

        json_start = completion_text.find("{")
        json_compatible_text = completion_text[json_start:].replace("'", "\"")

        benchmarks = json.loads(json_compatible_text)
        missing_fields = [key for key in expected_keys if key not in benchmarks]

        return benchmarks, missing_fields
    except json.JSONDecodeError as json_err:
        print(f"JSON decode error: {json_err}")
        return {key: {"N/A" : "bedrock"} for key in expected_keys}, expected_keys
    except Exception as e:
        print(f"Error fetching industry benchmarks: {e}")
        return {key: {"N/A" : "bedrock"} for key in expected_keys}, expected_keys


# Example Financial Health Summary
def financial_health_summary(df):
    required_fields = ["Total Revenue", "Total Net Income", "Total Expenses", "Debt-to-Equity Ratio"]
    metrics = {field: safe_access(df, field) for field in required_fields}
    
    # Handle cases where financial metrics are missing
    return {k: (v if v != "Not Present" else {"N/A": "report"}) for k, v in metrics.items()}

# Estimate Company Valuation
def estimate_company_valuation(df):
    financial_metrics = {
        "Total Revenue": safe_access(df, 'Total Revenue'),
        "Net Income": safe_access(df, 'Total Net Income'),
        "Total Expenses": safe_access(df, 'Total Expenses'),
        "Debt-to-Equity Ratio": safe_access(df, 'Debt-to-Equity Ratio'),
        "EBITDA": safe_access(df, 'Total Operating Income')
    }
    company_summary = safe_access(df, 'Summary')

    revenue_multiple = get_revenue_multiple_from_bedrock(company_summary, financial_metrics)  # Placeholder function
    if revenue_multiple is None:
        revenue_multiple = 5  # Default value if API call fails

    try:
        revenue = float(financial_metrics['Total Revenue'])
        valuation = revenue * revenue_multiple
    except (ValueError, TypeError):
        valuation = "Not Present"

    try:
        total_equity = float(safe_access(df, 'Total Equity'))
        market_to_book_ratio = valuation / total_equity if total_equity > 0 else 'N/A'
    except (ValueError, TypeError):
        market_to_book_ratio = 'N/A'

    return valuation, market_to_book_ratio


# Identify competitors using Amazon Bedrock
def get_competitors_from_bedrock(summary, retries=5, delay=1):
    model_id = "anthropic.claude-v2"
    prompt_text = (
        f"Human: Based on the following company description, identify a list of competitors and provide only their ticker symbols (e.g., AAPL, MSFT, GOOG).\n\n"
        f"Company Description: {summary}\n\nAssistant:"
    )
    payload = {
        "prompt": prompt_text,
        "max_tokens_to_sample": 1000,
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
        total_revenue = financials.loc['Total Revenue'].iloc[0] if 'Total Revenue' in financials.index else {"N/A": "web-search"}
        net_income = financials.loc['Net Income'].iloc[0] if 'Net Income' in financials.index else {"N/A": "web-search"}
        operating_income = financials.loc['Operating Income'].iloc[0] if 'Operating Income' in financials.index else {"N/A": "web-search"}
        ebitda = financials.loc['EBITDA'].iloc[0] if 'EBITDA' in financials.index else {"N/A": "web-search"}
        operating_cash_flow = cash_flow.loc['Total Cash From Operating Activities'].iloc[0] if 'Total Cash From Operating Activities' in cash_flow.index else {"N/A": "web-search"}

        # Calculating Debt-to-Equity Ratio
        total_liabilities = balance_sheet.loc['Total Liab'].iloc[0] if 'Total Liab' in balance_sheet.index else None
        total_equity = balance_sheet.loc['Total Stockholder Equity'].iloc[0] if 'Total Stockholder Equity' in balance_sheet.index else None
        debt_to_equity_ratio = (total_liabilities / total_equity) if total_liabilities and total_equity else {"N/A": "web-search"}

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



# Fetch industry expense ratios from Bedrock
def get_industry_expense_ratios_from_bedrock(company_summary):
    model_id = "anthropic.claude-v2"
    prompt_text = (
       f"Human: Provide industry averages for expenses as JSON with specific keys with the same format as the following format, replacing only the <value> fields and not adding any text, only return an answer in the format above without any modifications except for replacing the <value> fields:"
        f"{{'Industry COGS Ratio': <value>, 'Industry SG&A Ratio': <value>, 'Industry R&D Ratio': <value>, 'Industry Depreciation Ratio': <value>, 'Industry Interest Expense Ratio': <value>}}.\n\n"
        f"Company Description: {company_summary}\n\nAssistant:"
    )
    payload = {"prompt": prompt_text, "max_tokens_to_sample": 1000, "temperature": 0.5}
    expected_keys = ["Industry COGS Ratio", "Industry SG&A Ratio", "Industry R&D Ratio", "Industry Depreciation Ratio", "Industry Interest Expense Ratio"]

    try:
        response = client.invoke_model(modelId=model_id, body=json.dumps(payload), contentType="application/json")
        response_body = response['body'].read().decode()

        completion_text = json.loads(response_body).get('completion', '').strip()

        json_start = completion_text.find("{")
        json_compatible_text = completion_text[json_start:].replace("'", "\"")

        expense_ratios = json.loads(json_compatible_text)
        missing_fields = [key for key in expected_keys if key not in expense_ratios]

        return expense_ratios, missing_fields
    except json.JSONDecodeError as json_err:
        print(f"JSON decode error: {json_err}")
        return {key: {"N/A" : "bedrock"} for key in expected_keys}, expected_keys
    except Exception as e:
        print(f"Error fetching industry expense ratios: {e}")
        return {key: {"N/A" : "bedrock"} for key in expected_keys}, expected_keys


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
        f"Format your answer in bullet points. Your response should be of a maximum of 600 tokens and the last sentence should be complete and not cutoff.\n"
        f"Assistant:"
    )

    payload = {
        "prompt": prompt_text,
        "max_tokens_to_sample": 1000,
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

            # Process the analysis to create the desired dictionary structure
            if analysis:
                # Split the analysis into sentences and bullet points
                sentences = analysis.split('\n')
                first_sentence = sentences[0] if sentences else ""
                bullet_points = [sentence.strip() for sentence in sentences[1:] if sentence.strip()]

                # Create the dictionary structure
                summary_dict = {first_sentence: bullet_points}

                return summary_dict  # Return the summary dictionary

            return None  # In case of empty analysis

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
    
    prompt_text = (
        f"Human: Based on the following company profile and financial metrics, suggest a reasonable revenue multiple for valuation purposes. "
        f"Only provide the numeric value of the revenue multiple in your answer.\n\n"
        f"Company Description: {company_summary}\n\n"
        f"Financial Metrics: {financial_metrics}\n\n"
        f"Assistant:"
    )
    
    payload = {
        "prompt": prompt_text,
        "max_tokens_to_sample": 1000,
        "temperature": 0.7
    }

    for attempt in range(retries):
        try:
            response = client.invoke_model(
                modelId=model_id,
                body=json.dumps(payload),
                contentType="application/json"
            )
            response_body = response['body'].read().decode()

            multiple_text = json.loads(response_body).get('completion', '').strip()

            # Extract the numeric value
            value = re.search(r'\d+\.?\d*', multiple_text)  # Matches numeric values
            if value:
                return float(value.group(0))
            else:
                print("No numeric value found in response for revenue multiple.")
                return None

        except ValueError:
            print("The response did not contain a valid number for the revenue multiple.")
            return None
        except client.exceptions.ThrottlingException:
            if attempt < retries - 1:
                jitter = random.uniform(0, delay)
                wait_time = delay + jitter
                print(f"Throttling exception encountered. Retrying in {wait_time:.2f} seconds...")
                time.sleep(wait_time)
                delay *= 2
            else:
                print("Maximum retries reached. Exiting.")
                raise
        except Exception as e:
            print(f"An error occurred: {e}")
            return None


def mean_absolute_error(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate the absolute errors and return the mean
    return np.mean(np.abs(y_true - y_pred))


# Fetch industry revenue from Bedrock
def get_industry_revenue_from_bedrock(company_summary):
    model_id = "anthropic.claude-v2"
    prompt_text = (
        f"Human: Provide the total annual revenue of the industry as a JSON with the following format: "
        f"{{'Industry Revenue': <value>}}.\n\nCompany Description: {company_summary}\n\nAssistant:"
    )
    payload = {"prompt": prompt_text, "max_tokens_to_sample": 1000, "temperature": 0.5}
    expected_keys = ["Industry Revenue"]

    try:
        response = client.invoke_model(modelId=model_id, body=json.dumps(payload), contentType="application/json")
        response_body = response['body'].read().decode()

        completion_text = json.loads(response_body).get('completion', '').strip()

        json_start = completion_text.find("{")
        json_compatible_text = completion_text[json_start:].replace("'", "\"")

        industry_data = json.loads(json_compatible_text)
        return industry_data.get('Industry Revenue'), []

    except json.JSONDecodeError as json_err:
        print(f"JSON decode error: {json_err}")
        return {"N/A" : "bedrock"}, expected_keys
    except Exception as e:
        print(f"Error fetching industry revenue: {e}")
        return {"N/A" : "bedrock"}, expected_keys


# Fetch industry growth from Bedrock
def get_industry_growth_from_bedrock(company_summary):
    model_id = "anthropic.claude-v2"
    prompt_text = (
        f"Human: Provide the year-over-year growth rate for the industry as JSON with this format: "
        f"{{'Industry Growth Rate': <value>}}.\n\nCompany Description: {company_summary}\n\nAssistant:"
    )
    payload = {"prompt": prompt_text, "max_tokens_to_sample": 1000, "temperature": 0.5}
    expected_keys = ["Industry Growth Rate"]

    try:
        response = client.invoke_model(modelId=model_id, body=json.dumps(payload), contentType="application/json")
        response_body = response['body'].read().decode()

        completion_text = json.loads(response_body).get('completion', '').strip()

        json_start = completion_text.find("{")
        json_compatible_text = completion_text[json_start:].replace("'", "\"")

        industry_data = json.loads(json_compatible_text)
        return industry_data.get('Industry Growth Rate'), []

    except json.JSONDecodeError as json_err:
        print(f"JSON decode error: {json_err}")
        return {"N/A" : "bedrock"}, expected_keys
    except Exception as e:
        print(f"Error fetching industry growth rate: {e}")
        return {"N/A" : "bedrock"}, expected_keys



    
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
        "max_tokens_to_sample": 1000,
        "temperature": 0.3
    }
    
    for attempt in range(retries):
        try:
            response = client.invoke_model(modelId=model_id, body=json.dumps(payload), contentType="application/json")
            response_body = response['body'].read().decode()
            
            beta_text = json.loads(response_body).get('completion', '').strip()

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


def stock_price_prediction(company_financial_data, estimated_shares=1000000):
    try:
        # Ensure required fields are present
        if 'Total Revenue' not in company_financial_data or 'Total Net Income' not in company_financial_data:
            print("Missing required data for stock price prediction.")
            return "N/A", [], "N/A", "N/A"

        # Estimate valuation and market-to-book ratio using dynamic revenue multiple
        company_valuation, market_to_book_ratio = estimate_company_valuation(company_financial_data)

        # Use only the valuation (first element of the tuple) for IPO stock price calculation
        valuation = company_valuation

        # Updated data with valuation as target for prediction
        target_ipo_prices = [valuation]

        # Organize data into features for prediction
        data = {
            'Total_Revenue': [company_financial_data['Total Revenue'].iloc[0]],
            'Net_Income': [company_financial_data['Total Net Income'].iloc[0]],
            'Total_Expenses': [company_financial_data['Total Expenses'].iloc[0]],
            'Debt_to_Equity_Ratio': [company_financial_data['Debt-to-Equity Ratio'].iloc[0]],
            'EBITDA': [company_financial_data['Total Operating Income'].iloc[0]]
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

    except Exception as e:
        print(f"Error in stock price prediction: {e}")
        return "N/A", [], "N/A", "N/A"



def calculate_valuation_ratios(df):
    try:
        required_fields = ["Total Revenue", "Total Net Income", "Total Equity"]
        missing_fields = [field for field in required_fields if field not in df or pd.isna(df[field].iloc[0])]

        if missing_fields:
            return {"Error": f"Valuation Ratios are not computable because {', '.join(missing_fields)} are missing."}

        valuation_ratios = {}
        total_revenue = df['Total Revenue'].iloc[0]
        net_income = df['Total Net Income'].iloc[0]
        total_equity = df['Total Equity'].iloc[0]

        # Price-to-Earnings Ratio (P/E)
        valuation_ratios['Price-to-Earnings Ratio (P/E)'] = total_revenue / net_income if net_income else 'N/A'
        
        # Price-to-Sales Ratio (P/S)
        valuation_ratios['Price-to-Sales Ratio (P/S)'] = total_revenue / total_revenue if total_revenue else 'N/A'

        # Price-to-Book Ratio (P/B)
        valuation_ratios['Price-to-Book Ratio (P/B)'] = total_revenue / total_equity if total_equity else 'N/A'
        
        return valuation_ratios

    except Exception as e:
        print(f"Error calculating valuation ratios: {e}")
        return {"Error": "Unable to calculate valuation ratios"}




def calculate_risk_metrics_with_thresholds(df):
    try:
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
    
    except Exception as e:
        print(f"Error calculating risk metrics: {e}")
        return {"Error": "Unable to calculate risk metrics"}


def forecast_growth(df, growth_rate=0.05):
    try:
        required_fields = ["Total Revenue", "Total Net Income"]
        missing_fields = [field for field in required_fields if field not in df or pd.isna(df[field].iloc[0])]

        if missing_fields:
            return f"Growth Forecast is not computable because {', '.join(missing_fields)} are missing."

        forecasts = {}
        total_revenue = df['Total Revenue'].iloc[0]
        net_income = df['Total Net Income'].iloc[0]

        # Simple growth projections based on a static growth rate
        forecasts['Forecasted Revenue Growth'] = total_revenue * (1 + growth_rate)
        forecasts['Forecasted Net Income Growth'] = net_income * (1 + growth_rate)

        return forecasts

    except Exception as e:
        print(f"Error forecasting growth: {e}")
        return {"Error": "Unable to forecast growth"}


def get_industry_valuation_ratios_from_bedrock(company_summary, retries=5, delay=1):
    model_id = "anthropic.claude-v2"
    prompt_text = (
        f"Human: Based on the company description below, provide the average Price-to-Earnings (P/E) and Price-to-Sales (P/S) ratios for the industry.\n\n"
        f"Company Description: {company_summary}\n\nAssistant:"
    )
    payload = {
        "prompt": prompt_text,
        "max_tokens_to_sample": 10000,
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


def get_link_for_trend(category, retries=5, delay=1):
    """Fetch a relevant link for the given trend category from the model, returning only the link."""
    model_id = "anthropic.claude-v2"
    prompt_text = (
        f"Human: Provide a relevant online resource link for the following trend category without any additional text:\n\n"
        f"Trend Category: {category}\n\n"
        f"Assistant:"
    )
    payload = {
        "prompt": prompt_text,
        "max_tokens_to_sample": 1000,
        "temperature": 0.7
    }
    
    for attempt in range(retries):
        try:
            response = client.invoke_model(modelId=model_id, body=json.dumps(payload), contentType="application/json")
            response_body = json.loads(response['body'].read())
            link = response_body.get('completion', '').strip()
            return link  # Return only the fetched link

        except Exception as e:
            print(f"Error fetching link for trend category '{category}': {e}")
            if attempt < retries - 1:
                time.sleep(delay)
                delay *= 2
    return "N/A (Link not available)"  # Default return if fetching fails

def get_emerging_market_trends_from_bedrock(company_summary, retries=5, delay=1):
    model_id = "anthropic.claude-v2"
    prompt_text = (
        f"Human: Based on the company description below, list emerging market trends relevant to this company with a brief description for each trend:\n\n"
        f"Company Description: {company_summary}\n\n"
        f"Format your answer in bullet points. Your response should be of a maximum of 600 tokens and the last sentence should be complete and not cutoff.\n"
        f"Assistant:"
    )
    payload = {
        "prompt": prompt_text,
        "max_tokens_to_sample": 10000,
        "temperature": 0.7
    }
    
    for attempt in range(retries):
        try:
            response = client.invoke_model(modelId=model_id, body=json.dumps(payload), contentType="application/json")
            response_body = json.loads(response['body'].read())
            trends_text = response_body.get('completion', '').strip()
            
            # Process the trends text to create a structured output
            trends_lines = [trend.strip() for trend in trends_text.split('\n') if trend.strip()]
            first_trend_statement = "Here are some emerging market trends relevant to this company:"
            
            trends_list = []

            for line in trends_lines:
                if line.startswith('-'):
                    category_description = line[2:].strip()  # Remove the bullet point
                    try:
                        parts = category_description.split('-')
                        category = parts[0].strip()  # First part is the category name
                        dynamic_description = parts[1].strip() if len(parts) > 1 else "Description not available."
                        
                        # Fetch the link for the trend category
                        link = get_link_for_trend(category)  # Fetch link from model
                        
                        # Remove static text and just return the link
                        trends_list.append({
                            category: (dynamic_description, link.strip())
                        })
                    except IndexError:
                        print(f"Warning: Could not split category and description for line: {line}")
                        continue

            return {first_trend_statement: trends_list}  # Return the trends list

        except Exception as e:
            print(f"Error fetching emerging market trends: {e}")
            if attempt < retries - 1:
                time.sleep(delay)
                delay *= 2
    return None



# Call display_analysis function to print analyses
def fill_with_sources(data, source_dict):
    if isinstance(data, dict):  # Ensure data is a dict
        def get_value_or_placeholder(value, source):
            return value if value != "Not Present" else {"N/A": source}

        return {k: get_value_or_placeholder(v, source_dict.get(k, "Unknown Source")) for k, v in data.items()}
    else:
        return {"Error": "Invalid data format received."}


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

# Unpacking and organizing output in generate_analysis_json
def generate_analysis_json(input_json):
    # Transform the input JSON to the required format for analysis
    input_json = {
        'Report ID': ['Report_001'],  # Assuming a static report ID for example
        'Company Name': [input_json.get('Company Name', 'Unknown')],
        'Fiscal Year': [input_json.get('Fiscal Year')],
        'Report Date': [input_json.get('Report Date')],
        'Currency': [input_json.get('Currency', 'USD')],
        'Summary': [input_json.get('Summary', 'No summary available.')],
        'Total Revenue': [input_json.get('Total earnings')],  # Assuming 'Total earnings' is the correct key for revenue
        'Total Net Income': [input_json.get('Total Net Income')],
        'Total Operating Income': [input_json.get('Total Operating Income')],
        'Total Expenses': [input_json.get('Total Expenses')],
        'Cost of Goods Sold (COGS)': [input_json.get('Cost of Goods Sold (COGS)', 0)],
        'Selling, General, and Administrative (SG&A)': [input_json.get('Selling, General, and Administrative (SG&A)', 0)],
        'Research and Development (R&D)': [input_json.get('Research and Development (R&D)', 0)],
        'Depreciation and Amortization': [input_json.get('Depreciation and Amortization', 0)],
        'Interest Expense': [input_json.get('Interest Expense', 0)],
        'Other Expenses': [input_json.get('Other Expenses', 'Not Present')],
        'Total Debt': [input_json.get('Total Debt', 0)],
        'Debt-to-Equity Ratio': [input_json.get('Debt-to-Equity Ratio', 0)],
        'Long-Term Debt': [input_json.get('Long-Term Debt', 0)],
        'Short-Term Debt': [input_json.get('Short-Term Debt', 0)],
        'Total Equity': [input_json.get('Total Equity', 0)],
        'Gross Profit Margin': [input_json.get('Gross Profit Margin', 0)],
        'Operating Profit Margin': [input_json.get('Operating Profit Margin', 0)],
        'Net Profit Margin': [input_json.get('Net Profit Margin', 0)],
        'Return on Assets (ROA)': [input_json.get('Return on Assets (ROA)', 0)],
        'Return on Equity (ROE)': [input_json.get('Return on Equity (ROE)', 0)]
    }

    # Remove keys with null values
    report_data = {k: v for k, v in input_json.items() if v is not None}
    report_data = pd.DataFrame(report_data)

    # Mapping of each field to its source
    field_sources = {
        "Total Revenue": "report",
        "Total Net Income": "report",
        "Total Expenses": "report",
        "Debt-to-Equity Ratio": "report",
        "EBITDA": "report",
        "Industry Total Revenue": "bedrock",
        "Industry Net Income": "bedrock",
        "Industry Total Expenses": "bedrock",
        "Industry Debt-to-Equity Ratio": "bedrock",
        "Industry EBITDA": "bedrock",
        "COGS": "report",
        "SG&A": "report",
        "R&D": "report",
        "Depreciation and Amortization": "report",
        "Interest Expense": "report",
        "Other Expenses": "report",
        "Industry COGS Ratio": "bedrock",
        "Industry SG&A Ratio": "bedrock",
        "Industry R&D Ratio": "bedrock",
        "Industry Depreciation Ratio": "bedrock",
        "Industry Interest Expense Ratio": "bedrock"
    }

    # Fetch data and identify missing fields with sources
    industry_benchmarks, _ = get_industry_benchmarks(report_data['Summary'].iloc[0])
    industry_expenses, _ = get_industry_expense_ratios_from_bedrock(report_data['Summary'].iloc[0])
    industry_revenue, _ = get_industry_revenue_from_bedrock(report_data['Summary'].iloc[0])
    industry_growth, _ = get_industry_growth_from_bedrock(report_data['Summary'].iloc[0])
    beta_value, _ = get_beta_from_bedrock(report_data['Summary'].iloc[0])

    # Calculate company valuation, market-to-book ratio, and EPS
    valuation, market_to_book_ratio = estimate_company_valuation(report_data)
    estimated_shares = 1000000  # Example value
    eps = calculate_eps(report_data, estimated_shares)

    # Calculate risk metrics
    risk_metrics = calculate_risk_metrics_with_thresholds(report_data)

    # Forecast growth
    growth_forecast = forecast_growth(report_data)

    # Get industry valuation ratios
    industry_valuation_ratios = get_industry_valuation_ratios_from_bedrock(report_data['Summary'].iloc[0])

    # Get emerging market trends
    emerging_market_trends = get_emerging_market_trends_from_bedrock(report_data['Summary'].iloc[0])

    # Calculate stock price prediction to get MAE
    mae, predictions, company_valuation, ipo_stock_price = stock_price_prediction(estimated_shares)

    # Prepare data with sources indicated for missing fields
    output = {
        "Financial Health Summary": {
            "Metrics": financial_health_summary(report_data),
            "Industry Benchmarks": fill_with_sources(industry_benchmarks, field_sources),
            "Detailed Analysis": generate_summary_analysis("Financial Health Summary", financial_health_summary(report_data), {})
        },
        "Expense Breakdown": {
            "Expenses": expense_breakdown(report_data),
            "Industry Expense Ratios": fill_with_sources(industry_expenses, field_sources),
            "Detailed Analysis": generate_summary_analysis("Expense Breakdown", expense_breakdown(report_data), {})
        },
        "Competitor Comparison": {
            "Competitors Data": convert_to_serializable(competitor_comparison(report_data)),
            "Detailed Analysis": generate_summary_analysis("Competitor Comparison", competitor_comparison(report_data).to_string(index=False), {})
        },
        "Company Valuation": {
            "Estimated Valuation": f"${valuation:,.2f}" if isinstance(valuation, (int, float)) else {"N/A": "bedrock"},
            "Market-to-Book Ratio": f"{market_to_book_ratio:.2f}" if isinstance(market_to_book_ratio, (int, float)) else {"N/A": "bedrock"},
            "Detailed Analysis": generate_summary_analysis("Company Valuation", f"Estimated Valuation: ${valuation:,.2f}" if isinstance(valuation, (int, float)) else "N/A", {})
        },
        "Earnings Per Share (EPS)": {
            "EPS": f"${eps:.2f}" if isinstance(eps, (int, float)) else {"N/A": "bedrock"},
            "Detailed Analysis": generate_summary_analysis("Earnings Per Share", f"Earnings Per Share (EPS): ${eps:.2f}" if isinstance(eps, (int, float)) else "N/A", {})
        },
        "Market Analysis": {
            "Market Share": calculate_market_share(report_data['Total Revenue'].iloc[0], industry_revenue) if 'Total Revenue' in report_data and isinstance(calculate_market_share(report_data['Total Revenue'].iloc[0], industry_revenue), (int, float)) else {"N/A": "bedrock"},
            "Industry Revenue Growth": industry_growth if isinstance(industry_growth, (int, float)) else {"N/A": "bedrock"},
            "Beta (Volatility)": beta_value if isinstance(beta_value, (int, float)) else {"N/A": "bedrock"}
        },
        "Valuation Ratios": {
            "Valuation Ratios": fill_with_sources(calculate_valuation_ratios(report_data), field_sources),
            "Detailed Analysis": generate_summary_analysis("Valuation Ratios", calculate_valuation_ratios(report_data), {})
        },
        "Risk Metrics": {
            "Risk Metrics": fill_with_sources(risk_metrics, field_sources),
            "Detailed Analysis": generate_summary_analysis("Risk Metrics", risk_metrics, {})
        },
        "Growth Forecast": {
            "Forecasts": fill_with_sources(growth_forecast, field_sources),
            "Detailed Analysis": generate_summary_analysis("Growth Forecast", growth_forecast, {})
        },
        "Emerging Market Trends": {
            "Trends": emerging_market_trends if emerging_market_trends else {"N/A": "bedrock"},
            "Detailed Analysis": generate_summary_analysis("Emerging Market Trends", emerging_market_trends, {})
        },
        "Estimated Company Valuation Summary": {
            "Final Estimated Valuation": f"${valuation:,.2f}" if isinstance(valuation, (int, float)) else {"N/A": "bedrock"},
            "Mean Absolute Error": f"{mae:.2f}" if isinstance(mae, (int, float)) else {"N/A": "bedrock"},
            "Predicted IPO Valuation": f"${predictions[0]:,.2f}" if isinstance(predictions, list) and predictions else {"N/A": "bedrock"},
            "Estimated IPO Stock Price per Share": f"${ipo_stock_price:.2f}" if isinstance(ipo_stock_price, (int, float)) else {"N/A": "bedrock"}
        }
    }
    
    return json.dumps(convert_to_serializable(output), indent=4)

# Call generate_analysis_json to create JSON output with source-indicated placeholders

if __name__ == "__main__":
    input_json = {
        'Company Name': 'PrivateTech',
        'Fiscal Year': 2022,
        'Report Date': '2022-12-31',
        'Currency': 'USD',
        'Summary': 'PrivateTech develops software solutions for e-commerce optimization in North America.',
        'Total earnings': 5006780,
        'Total Net Income': 707800,
        'Total Operating Income': 609800,
        'Total Expenses': 4309700,
        'Cost of Goods Sold (COGS)': 2500560,
        'Selling, General, and Administrative (SG&A)': 1225000,
        'Research and Development (R&D)': 400180,
        'Depreciation and Amortization': 200000,
        'Interest Expense': 100000,
        'Other Expenses': None,  # This will not be included
        'Total Debt': 2000000,
        'Debt-to-Equity Ratio': 0.5,
        'Long-Term Debt': 1500000,
        'Short-Term Debt': 500000,
        'Total Equity': 4000000,
        'Gross Profit Margin': 0.55,
        'Operating Profit Margin': 0.12,
        'Net Profit Margin': 0.14,
        'Return on Assets (ROA)': 0.10,
        'Return on Equity (ROE)': 0.175
    }
    json_output = generate_analysis_json(input_json)
    print("=== JSON Output for UI ===")
    print(json_output)
