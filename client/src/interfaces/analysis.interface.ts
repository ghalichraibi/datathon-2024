interface FinancialHealthSummary {
  TotalRevenue: number | null;
  TotalNetIncome: number | null;
  TotalExpenses: number | null;
  DebtToEquityRatio: number | null;
  EBITDA: number | null;
  IndustryBenchmarks: {
    IndustryTotalRevenue: number | null;
    IndustryNetIncome: number | null;
    IndustryTotalExpenses: number | null;
    IndustryDebtToEquityRatio: number | null;
    IndustryEBITDA: number | null;
  };
  DetailedAnalysis: string[];
}

interface ExpenseBreakdown {
  Expenses: {
    COGS: number | null;
    SGA: number | null;
    RandD: number | null;
    DepreciationAndAmortization: number | null;
    InterestExpense: number | null;
    OtherExpenses: number | null;
  };
  ExpenseRatios: {
    COGS: number | null;
    SGA: number | null;
    RandD: number | null;
    DepreciationAndAmortization: number | null;
    InterestExpense: number | null;
    OtherExpenses: number | null;
  };
  DetailedAnalysis: string[];
}

interface CompetitorData {
  Symbol: string | null;
  TotalRevenue: number | null;
  NetIncome: number | null;
  OperatingIncome: number | null;
  DebtToEquityRatio: number | null;
  EBITDA: number | null;
  CashFlow: number | null;
}

interface CompetitorComparison {
  CompetitorData: CompetitorData[];
  DetailedAnalysis: string[];
}

interface CompanyValuation {
  EstimatedValuation: number | null;
  MarketToBookRatio: number | null;
  DetailedAnalysis: string[];
}

interface EPS {
  EPSValue: number | null;
  DetailedAnalysis: string[];
}

interface ValuationRatios {
  PE: number | null;
  PS: number | null;
  PB: number | null;
  DetailedAnalysis: string[];
}

interface RiskMetrics {
  DebtToEquityRatio: number | null;
  ProfitMargin: number | null;
  DebtToEquityRiskCategory: string | null;
  ProfitMarginRiskCategory: string | null;
  DetailedAnalysis: string[];
}

interface GrowthForecast {
  ForecastedRevenueGrowth: number | null;
  ForecastedNetIncomeGrowth: number | null;
  DetailedAnalysis: string[];
}

interface EmergingTrend {
  Category: string;
  Details: string[];
}

interface MarketAnalysis {
  MarketShare: any;
  IndustryRevenueGrowth: any;
  Beta: number | null;
  EmergingTrends: EmergingTrend[];
  DetailedAnalysis: string[];
}

interface ValuationSummary {
  FinalValuation: number | null;
  MeanAbsoluteError: any;
  PredictedIPOValuation: any;
  IPOStockPrice: any;
  DetailedAnalysis: string[];
}

export interface Analysis {
  FinancialHealthSummary: FinancialHealthSummary;
  ExpenseBreakdown: ExpenseBreakdown;
  CompetitorComparison: CompetitorComparison;
  CompanyValuation: CompanyValuation;
  EPS: EPS;
  ValuationRatios: ValuationRatios;
  RiskMetrics: RiskMetrics;
  GrowthForecast: GrowthForecast;
  MarketAnalysis: MarketAnalysis;
  ValuationSummary: ValuationSummary;
}
