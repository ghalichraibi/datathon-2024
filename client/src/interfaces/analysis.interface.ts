interface FinancialHealthSummary {
  TotalRevenue: number | null;
  TotalNetIncome: number | null;
  TotalExpenses: number | null;
  DebtToEquityRatio: number | null;
  EBITDA: number | null;
  DetailedAnalysis: string[];
}

interface ExpenseBreakdown {
  COGS: number | null;
  SGA: number | null;
  RandD: number | null;
  DepreciationAndAmortization: number | null;
  InterestExpense: number | null;
  OtherExpenses: number | null;
  TotalExpenses: number | null;
  ExpenseRatios: {
    COGS: number | null;
    SGA: number | null;
    RandD: number | null;
    DepreciationAndAmortization: number | null;
    InterestExpense: number | null;
    OtherExpenses: string | null;
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
  CompetitorComparison: CompetitorData[];
  DetailedAnalysis: string[];
}

interface IndustryMetrics {
  RevenueGrowth: number | null;
  ValuationRatios: {
    PE: number | null;
    PS: number | null;
  };
  Beta: number | null;
}

interface MarketAnalysis {
  Industry: IndustryMetrics;
  EmergingTrends: string[];
  DetailedAnalysis: string[];
  MissingDataExplanation: string;
}

interface ValuationSummary {
  FinalValuation: number | null;
  IPOStockPrice: number | null;
  MarketShare: number | null;
  MeanAbsoluteError: number | null;
  PredictedIPOValuation: number | null;
  DetailedAnalysis: string[];
}

export interface Analysis {
  FinancialHealthSummary: FinancialHealthSummary;
  ExpenseBreakdown: ExpenseBreakdown;
  CompetitorComparison: CompetitorComparison;
  CompanyValuation: CompanyValuation;
  EPS: EPS;
  MarketAnalysis: MarketAnalysis;
  ValuationSummary: ValuationSummary;
}
