import { Analysis } from 'interfaces/analysis.interface';

export class AnalysisService {
  public static analysis: Analysis;

  private static baseUrl = 'http://127.0.0.1:5000';

  public static async analyzeReport(reports: File[]): Promise<Analysis> {
    this.validateFiles(reports);

    try {
      // TODO: implement the logic to analyze multiple reports
      const mostRecentReport = this.selectMostRecentReport(reports);

      const formData = new FormData();
      formData.append('pdf', mostRecentReport);

      const response = await fetch(`${this.baseUrl}/analyze`, {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        throw new Error('Failed to analyze report');
      }

      this.analysis = await response.json();
      return this.analysis;
    } catch (error) {
      throw new Error('Failed to analyze report');
    }
  }

  public static setAnalysis(analysis: Analysis): void {
    this.analysis = analysis;
  }

  public static getAnalysis(): Analysis {
    // return a deep copy of the analysis
    return JSON.parse(JSON.stringify(this.analysis));
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
