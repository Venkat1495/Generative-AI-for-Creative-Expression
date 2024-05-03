import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})



export class PredictionService {
  private apiUrl = 'http://127.0.0.1';

  constructor(private http: HttpClient) {}

  // Get history
  getHistory(): Observable<any> {
    return this.http.get(`${this.apiUrl}/history`);
  }

  // Submit prediction request
  submitPrediction(predictionRequest: any): Observable<any> {
    return this.http.post(`${this.apiUrl}/prediction`, predictionRequest);
  }

  // Get specific prediction
  getSpecificPrediction(urlSegment: string): Observable<any> {
    return this.http.get(`${this.apiUrl}/prediction/${urlSegment}`);
  }
}
