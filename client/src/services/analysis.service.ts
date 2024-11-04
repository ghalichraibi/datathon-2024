import { Analysis } from 'interfaces/analysis.interface';


window.onerror = function (message, source, lineno, colno, error) {
  console.log('Error details:', { message, source, lineno, colno, error });
};

export class AnalysisService {
  private static analysis: Analysis;
  private static baseUrl = import.meta.env.VITE_API_URL;

  public static async analyzeReport(reports: File[]): Promise<Analysis> {
    this.validateFiles(reports);

    try {
      const mostRecentReport = this.selectMostRecentReport(reports);

      const formData = new FormData();
      formData.append('pdf', mostRecentReport);

      for (let [key, value] of formData.entries()) {
        console.log(`${key}:`, value);
      }


      const response = await fetch(`${this.baseUrl}/analyze`, {
        method: 'POST',
        body: formData
      });

      console.log('API URL:', this.baseUrl); // Ensure it's correctly pointing to http://127.0.0.1:5000 or similar

      if (!response.ok) {
        // Log the response status and text for debugging
        const errorText = await response.text();
        console.error(`Failed to analyze report: ${response.status} - ${errorText}`);
        throw new Error(`Failed to analyze report: ${response.status} - ${errorText}`);
      }

      this.analysis = await response.json();
      return this.analysis;
    } catch (error) {
      console.error('An error occurred while analyzing the report:', error);
      throw new Error('Failed to analyze report');
    }
  }

  public static convertJsonToAnalysis(json: any): Analysis {
    const analysis: Analysis = {
      FinancialHealthSummary: {
        TotalRevenue: null,
        TotalNetIncome: null,
        TotalExpenses: null,
        DebtToEquityRatio: null,
        EBITDA: null,
        IndustryBenchmarks: {
          IndustryTotalRevenue: null,
          IndustryNetIncome: null,
          IndustryTotalExpenses: null,
          IndustryDebtToEquityRatio: null,
          IndustryEBITDA: null
        },
        DetailedAnalysis: []
      },
      ExpenseBreakdown: {
        Expenses: {
          COGS: null,
          SGA: null,
          RandD: null,
          DepreciationAndAmortization: null,
          InterestExpense: null,
          OtherExpenses: null
        },
        ExpenseRatios: {
          COGS: null,
          SGA: null,
          RandD: null,
          DepreciationAndAmortization: null,
          InterestExpense: null,
          OtherExpenses: null
        },
        DetailedAnalysis: []
      },
      CompetitorComparison: {
        CompetitorData: [],
        DetailedAnalysis: []
      },
      CompanyValuation: {
        EstimatedValuation: null,
        MarketToBookRatio: null,
        DetailedAnalysis: []
      },
      EPS: {
        EPSValue: null,
        DetailedAnalysis: []
      },
      ValuationRatios: {
        PE: null,
        PS: null,
        PB: null,
        DetailedAnalysis: []
      },
      RiskMetrics: {
        DebtToEquityRatio: null,
        ProfitMargin: null,
        DebtToEquityRiskCategory: null,
        ProfitMarginRiskCategory: null,
        DetailedAnalysis: []
      },
      GrowthForecast: {
        ForecastedRevenueGrowth: null,
        ForecastedNetIncomeGrowth: null,
        DetailedAnalysis: []
      },
      MarketAnalysis: {
        MarketShare: null,
        IndustryRevenueGrowth: null,
        Beta: null,
        EmergingTrends: [],
        DetailedAnalysis: []
      },
      ValuationSummary: {
        FinalValuation: null,
        MeanAbsoluteError: null,
        PredictedIPOValuation: null,
        IPOStockPrice: null,
        DetailedAnalysis: []
      }
    };

    // Helper function to safely get values
    const getValue = (value: any): any => {
      return value !== undefined && value !== null ? value : null;
    };

    // Helper function to parse numbers from strings with currency symbols
    const parseNumber = (value: any): number | null => {
      if (typeof value === 'number') return value;
      if (typeof value === 'string') {
        const parsedValue = parseFloat(value.replace(/[\$,]/g, ''));
        return isNaN(parsedValue) ? null : parsedValue;
      }
      return null;
    };

    // Financial Health Summary
    if (json['Financial Health Summary']) {
      const fhs = json['Financial Health Summary'];
      const metrics = fhs.Metrics || {};

      analysis.FinancialHealthSummary.TotalRevenue = parseNumber(metrics['Total Revenue']);
      analysis.FinancialHealthSummary.TotalNetIncome = parseNumber(metrics['Total Net Income']);
      analysis.FinancialHealthSummary.TotalExpenses = parseNumber(metrics['Total Expenses']);
      analysis.FinancialHealthSummary.DebtToEquityRatio = parseNumber(metrics['Debt-to-Equity Ratio']);
      analysis.FinancialHealthSummary.EBITDA = parseNumber(metrics['EBITDA']);

      const industryBenchmarks = fhs['Industry Benchmarks'] || {};
      analysis.FinancialHealthSummary.IndustryBenchmarks = {
        IndustryTotalRevenue: parseNumber(industryBenchmarks['Industry Total Revenue']),
        IndustryNetIncome: parseNumber(industryBenchmarks['Industry Net Income']),
        IndustryTotalExpenses: parseNumber(industryBenchmarks['Industry Total Expenses']),
        IndustryDebtToEquityRatio: parseNumber(industryBenchmarks['Industry Debt-to-Equity Ratio']),
        IndustryEBITDA: parseNumber(industryBenchmarks['Industry EBITDA'])
      };

      const detailedAnalysisObj = fhs['Detailed Analysis'] || {};
      for (const key in detailedAnalysisObj) {
        const arr = detailedAnalysisObj[key];
        if (Array.isArray(arr)) {
          analysis.FinancialHealthSummary.DetailedAnalysis.push(...arr);
        }
      }
    }

    // Expense Breakdown
    if (json['Expense Breakdown']) {
      const eb = json['Expense Breakdown'];
      const expenses = eb['Expenses'] || {};

      analysis.ExpenseBreakdown.Expenses = {
        COGS: parseNumber(expenses['COGS']),
        SGA: parseNumber(expenses['SG&A']),
        RandD: parseNumber(expenses['R&D']),
        DepreciationAndAmortization: parseNumber(expenses['Depreciation and Amortization']),
        InterestExpense: parseNumber(expenses['Interest Expense']),
        OtherExpenses: parseNumber(expenses['Other Expenses'])
      };

      const industryExpenseRatios = eb['Industry Expense Ratios'] || {};
      analysis.ExpenseBreakdown.ExpenseRatios = {
        COGS: getValue(industryExpenseRatios['Industry COGS Ratio']),
        SGA: getValue(industryExpenseRatios['Industry SG&A Ratio']),
        RandD: getValue(industryExpenseRatios['Industry R&D Ratio']),
        DepreciationAndAmortization: getValue(industryExpenseRatios['Industry Depreciation Ratio']),
        InterestExpense: getValue(industryExpenseRatios['Industry Interest Expense Ratio']),
        OtherExpenses: getValue(industryExpenseRatios['Industry Other Expenses Ratio'])
      };

      const detailedAnalysisObj = eb['Detailed Analysis'] || {};
      for (const key in detailedAnalysisObj) {
        const arr = detailedAnalysisObj[key];
        if (Array.isArray(arr)) {
          analysis.ExpenseBreakdown.DetailedAnalysis.push(...arr);
        }
      }
    }

    // Competitor Comparison
    if (json['Competitor Comparison']) {
      const cc = json['Competitor Comparison'];
      const competitors = cc['Competitors Data'] || [];

      analysis.CompetitorComparison.CompetitorData = competitors.map((comp: any) => ({
        Symbol: getValue(comp.Symbol),
        TotalRevenue: parseNumber(comp['Total Revenue']),
        NetIncome: parseNumber(comp['Net Income']),
        OperatingIncome: parseNumber(comp['Operating Income']),
        DebtToEquityRatio: parseNumber(comp['Debt-to-Equity Ratio']),
        EBITDA: parseNumber(comp['EBITDA']),
        CashFlow: parseNumber(comp['Cash Flow'])
      }));

      const detailedAnalysisObj = cc['Detailed Analysis'] || {};
      for (const key in detailedAnalysisObj) {
        const arr = detailedAnalysisObj[key];
        if (Array.isArray(arr)) {
          analysis.CompetitorComparison.DetailedAnalysis.push(...arr);
        }
      }
    }

    // Company Valuation
    if (json['Company Valuation']) {
      const cv = json['Company Valuation'];

      analysis.CompanyValuation.EstimatedValuation = parseNumber(cv['Estimated Valuation']);
      analysis.CompanyValuation.MarketToBookRatio = parseNumber(cv['Market-to-Book Ratio']);

      const detailedAnalysisObj = cv['Detailed Analysis'] || {};
      for (const key in detailedAnalysisObj) {
        const arr = detailedAnalysisObj[key];
        if (Array.isArray(arr)) {
          analysis.CompanyValuation.DetailedAnalysis.push(...arr);
        }
      }
    }

    // EPS
    if (json['Earnings Per Share (EPS)']) {
      const eps = json['Earnings Per Share (EPS)'];

      analysis.EPS.EPSValue = parseNumber(eps['EPS']);

      const detailedAnalysisObj = eps['Detailed Analysis'] || {};
      for (const key in detailedAnalysisObj) {
        const arr = detailedAnalysisObj[key];
        if (Array.isArray(arr)) {
          analysis.EPS.DetailedAnalysis.push(...arr);
        }
      }
    }

    // Valuation Ratios
    if (json['Valuation Ratios']) {
      const vr = json['Valuation Ratios'];
      const ratios = vr['Valuation Ratios'] || {};

      analysis.ValuationRatios.PE = parseNumber(ratios['Price-to-Earnings Ratio (P/E)']);
      analysis.ValuationRatios.PS = parseNumber(ratios['Price-to-Sales Ratio (P/S)']);
      analysis.ValuationRatios.PB = parseNumber(ratios['Price-to-Book Ratio (P/B)']);

      const detailedAnalysisObj = vr['Detailed Analysis'] || {};
      for (const key in detailedAnalysisObj) {
        const arr = detailedAnalysisObj[key];
        if (Array.isArray(arr)) {
          analysis.ValuationRatios.DetailedAnalysis.push(...arr);
        }
      }
    }

    // Risk Metrics
    if (json['Risk Metrics']) {
      const rm = json['Risk Metrics'];
      const metrics = rm['Risk Metrics'] || {};

      analysis.RiskMetrics.DebtToEquityRatio = parseNumber(metrics['Debt-to-Equity Ratio']);
      analysis.RiskMetrics.ProfitMargin = parseNumber(metrics['Profit Margin']);
      analysis.RiskMetrics.DebtToEquityRiskCategory = getValue(metrics['Debt-to-Equity Ratio Risk Category']);
      analysis.RiskMetrics.ProfitMarginRiskCategory = getValue(metrics['Profit Margin Risk Category']);

      const detailedAnalysisObj = rm['Detailed Analysis'] || {};
      for (const key in detailedAnalysisObj) {
        const arr = detailedAnalysisObj[key];
        if (Array.isArray(arr)) {
          analysis.RiskMetrics.DetailedAnalysis.push(...arr);
        }
      }
    }

    // Growth Forecast
    if (json['Growth Forecast']) {
      const gf = json['Growth Forecast'];
      const forecasts = gf['Forecasts'] || {};

      analysis.GrowthForecast.ForecastedRevenueGrowth = parseNumber(forecasts['Forecasted Revenue Growth']);
      analysis.GrowthForecast.ForecastedNetIncomeGrowth = parseNumber(forecasts['Forecasted Net Income Growth']);

      const detailedAnalysisObj = gf['Detailed Analysis'] || {};
      for (const key in detailedAnalysisObj) {
        const arr = detailedAnalysisObj[key];
        if (Array.isArray(arr)) {
          analysis.GrowthForecast.DetailedAnalysis.push(...arr);
        }
      }
    }

    // Market Analysis
    if (json['Market Analysis']) {
      const ma = json['Market Analysis'];

      analysis.MarketAnalysis.MarketShare = getValue(ma['Market Share']);
      analysis.MarketAnalysis.IndustryRevenueGrowth = getValue(ma['Industry Revenue Growth']);
      analysis.MarketAnalysis.Beta = parseNumber(ma['Beta (Volatility)']);
    }

    // Emerging Market Trends
    if (json['Emerging Market Trends']) {
      const trendsObj = json['Emerging Market Trends']['Trends'] || {};
      const trendsArray = trendsObj['Here are some emerging market trends relevant to this company:'] || [];

      analysis.MarketAnalysis.EmergingTrends = trendsArray.map((trendItem: any) => {
        const category = Object.keys(trendItem)[0];
        const details = trendItem[category];
        return {
          Category: category,
          Details: details
        };
      });

      const detailedAnalysisObj = json['Emerging Market Trends']['Detailed Analysis'] || {};
      for (const key in detailedAnalysisObj) {
        const arr = detailedAnalysisObj[key];
        if (Array.isArray(arr)) {
          analysis.MarketAnalysis.DetailedAnalysis.push(...arr);
        }
      }
    }

    // Valuation Summary
    if (json['Estimated Company Valuation Summary']) {
      const vs = json['Estimated Company Valuation Summary'];

      analysis.ValuationSummary.FinalValuation = parseNumber(vs['Final Estimated Valuation']);
      analysis.ValuationSummary.MeanAbsoluteError = getValue(vs['Mean Absolute Error']);
      analysis.ValuationSummary.PredictedIPOValuation = getValue(vs['Predicted IPO Valuation']);
      analysis.ValuationSummary.IPOStockPrice = getValue(vs['Estimated IPO Stock Price per Share']);
    }

    return analysis;
  }

  public static setAnalysis(analysis: Analysis): void {
    this.analysis = analysis;
  }

  public static getAnalysis(): Analysis | null {
    return this.analysis ? JSON.parse(JSON.stringify(this.analysis)) : null;
  }

  private static selectMostRecentReport(files: File[]): File {
    const yearRegex = /20\d{2}/;
    let mostRecentFile = files[0];
    let mostRecentYear = 0;

    for (const file of files) {
      const match = file.name.match(yearRegex);
      if (match) {
        const year = parseInt(match[0]);
        if (year > mostRecentYear) {
          mostRecentYear = year;
          mostRecentFile = file;
        }
      }
    }

    return mostRecentFile;
  }

  private static validateFiles(files: File[]): void {
    if (!files || !files.length) {
      throw new Error('No files provided');
    }

    for (const file of files) {
      if (!file || !file.name || !file.size || file.type !== 'application/pdf') {
        throw new Error('Invalid file provided');
      }
    }
  }
}
