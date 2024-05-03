import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { HttpClientModule, HttpClient, HttpHeaders } from '@angular/common/http';
import { Observable, throwError } from 'rxjs';
import { Router } from '@angular/router';

@Component({
  selector: 'app-prediction',
  standalone: true,
  imports: [CommonModule, FormsModule, HttpClientModule], // Ensure all necessary modules are imported
  templateUrl: './prediction.component.html',
  styleUrls: ['./prediction.component.scss']
})
export class PredictionComponent implements OnInit {
  historyList: Array<{ title: string; urlSegment: string }> = [];
  selectedUrlSegment: string = '';
  // predictionResults: any = null;
  isLoading: boolean = false;

  private apiUrl = 'http://127.0.0.1:8000';

  predictionRequest = {
    Title: '',
    Genre: '',
    Artist: '',
    Number_of_Samples: 0
  };

  predictionResults = {
    Title: '',
    Genre: '',
    Artist: '',
    Number_of_Samples: 0,
    Lyrics: '',
    GPT_Lyrics: ''
  };


  constructor(private http: HttpClient,  private router: Router) {}

  ngOnInit(): void {
    this.loadHistory();
  }

  loadHistory(): void {
    this.getHistory().subscribe({
      next: (data) => {this.historyList = data; console.log("Geting History:", data);
      if (this.historyList.length > 0) {
        // Automatically select the last item in the history list
        this.onSelectTitle(this.historyList[0]);
      }
      },
      error: (error) => {
        console.error('Error loading history:', error);
        if (error.status === 401) {
          // Redirect to login page on Unauthorized error
          this.router.navigate(['/login']);
        }
      }
    });
  }

  onSubmit(): void {
    this.isLoading = true; // Start loading
    this.submitPrediction(this.predictionRequest).subscribe({
      next: (data) => {console.log("New Prediction is suceessful:", data["message"]);
      this.loadHistory();
      this.isLoading = false; // Stop loading
      },
      error: (error) => {
        console.error('Error submitting prediction:', error);
        if (error.status === 401) {
          // Redirect to login page on Unauthorized error
          this.router.navigate(['/login']);
        }
      }
    });
  }

  onSelectTitle(item: { title: string; urlSegment: string }): void {
    this.selectedUrlSegment = item.urlSegment;
    this.getSpecificPrediction(item.urlSegment).subscribe({
      next: (data) => {
        this.predictionResults = data;
        console.log("Getting Old Prediction is successful:", data);
      },
      error: (error) => {
        console.error('Error getting specific prediction:', error);
        if (error.status === 401) {
          this.router.navigate(['/login']);
        }
      }
    });
  }

  onLogout(): void {
    localStorage.removeItem('access_token'); // Clear the authentication token
    this.router.navigate(['/login']); // Navigate to login page
    console.log("Logged out successfully.");
  }

  // Utility method to get headers
  private getHeaders(): HttpHeaders {
    const token = localStorage.getItem('access_token');
    if (!token) {
      console.error('No access token found. Redirecting to login.');
      this.router.navigate(['/login']); // Redirect to login page
      // return null; // Return null if no token found
    }
    return new HttpHeaders({
      'accept': 'application/json',
      'Authorization': `Bearer ${token}`
    });
  }

  // Revised HTTP methods using the utility method for headers
  getHistory(): Observable<any> {
    const headers = this.getHeaders();
    if (!headers) return throwError(() => new Error('No headers available'));
    return this.http.get(`${this.apiUrl}/get_history`, { headers });
  }

  submitPrediction(predictionRequest: any): Observable<any> {
    const headers = this.getHeaders();
    if (!headers) return throwError(() => new Error('No headers available'));
    return this.http.post(`${this.apiUrl}/prediction`, predictionRequest, { headers });
  }

  getSpecificPrediction(urlSegment: string): Observable<any> {
    const headers = this.getHeaders();
    if (!headers) return throwError(() => new Error('No headers available'));
    return this.http.get(`${this.apiUrl}/prediction/${urlSegment}`, { headers });
  }
}


// With Service have to fix or study later


// import { Component, OnInit } from '@angular/core';
// import { HttpClientModule, HttpClient } from '@angular/common/http';
// import { PredictionService } from './prediction.service';
// import { CommonModule } from '@angular/common';
// import { FormsModule } from '@angular/forms';


// @Component({
//   selector: 'app-prediction',
//   standalone: true,
//   imports: [CommonModule, FormsModule, HttpClientModule],
//   templateUrl: './prediction.component.html',
//   styleUrl: './prediction.component.scss'
// })

// export class PredictionComponent implements OnInit {
//   historyList: Array<{ title: string; link: string }> = [];
//   selectedTitle: string = '';
//   predictionResults: any = null;

//   predictionRequest = {
//     Title: '',
//     Genre: '',
//     Artist: '',
//     Number_of_Samples: 0
//   };

//   constructor(
//     private http: HttpClient,
//     private predictionService: PredictionService) {}

//   ngOnInit(): void {
//     this.loadHistory();
//   }

//   loadHistory(): void {
//     this.predictionService.getHistory().subscribe(data => {
//       this.historyList = data;
//     });
//   }

//   onSubmit(): void {
//     this.predictionService.submitPrediction(this.predictionRequest).subscribe(data => {
//       this.predictionResults = data;
//     });
//   }

//   onSelectTitle(title: string, link: string): void {
//     this.selectedTitle = title;
//     this.predictionService.getSpecificPrediction(link).subscribe(data => {
//       this.predictionResults = data;
//     });
//   }
// }
