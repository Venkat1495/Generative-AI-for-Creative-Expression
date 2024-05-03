import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { Router } from '@angular/router';  // Import Router

@Component({
  selector: 'app-login',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './login.component.html',
  styleUrls: ['./login.component.scss']
})
export class LoginComponent {
  loginobj = new Loginveb();
  signupobj = new signUpveb();
  errorMessage = '';  // Initialize to an empty string

  constructor(private http: HttpClient, private router: Router) {  // Inject HttpClient and Router

  }

  onLogin(): void {
    const body = new URLSearchParams();
    body.set('username', this.loginobj.username);
    body.set('password', this.loginobj.password);

    let headers = new HttpHeaders({
      'accept': 'application/json',
      'Content-Type': 'application/x-www-form-urlencoded'
    });

    this.http.post<{access_token: string, token_type: string}>('http://127.0.0.1:8000/login', body.toString(), { headers: headers }).subscribe({
      next: (response) => {
        console.log('Login successful', response);
        
        localStorage.setItem('access_token', response.access_token); // Store the token in localStorage
        this.router.navigate(['/prediction']); // Navigate to the prediction page
      },
      error: (error) => {
        console.error('Login error:', error);
        this.errorMessage = error.statusText || 'Failed to login';
      }
    });
  }

  onSignUp(): void {
    this.http.post('http://127.0.0.1:8000/users', this.signupobj).subscribe({
      next: (response) => {
        console.log('Signup successful', response);
        this.signUp('signUp')
        this.router.navigate(['/login']); // Navigate back to the login page after successful signup
      },
      error: (error) => {
        console.error('Signup error:', error);
        this.errorMessage = error.message || 'An error occurred during signup.';
      }
    });
  }

  isApproved = false

  signUp(status: string): void {
    this.isApproved = status === "login";
  }

  continueClicked() {
    console.log("Continue Clicked !!");
  }
}

export class Loginveb {
  username = '';
  password = '';
}

export class signUpveb {
  email = '';
  password = '';
  first_name = '';
  last_name = '';
}



// --------- Trying and fixing GPT code above below is backup

// import { HttpClient, HttpClientModule, HttpHeaders } from '@angular/common/http';
// import { CommonModule } from '@angular/common'; // Import CommonModule
// import { Component } from '@angular/core';
// import { FormsModule } from '@angular/forms';

// @Component({
//   selector: 'app-login',
//   standalone: true,
//   imports: [CommonModule, HttpClientModule, FormsModule],
//   templateUrl: './login.component.html',
//   styleUrls: ['./login.component.scss']
// })
// export class LoginComponent {
  
//   loginobj : Loginveb;
//   signupobj : signUpveb;
//   errorMessage: string; // For displaying error messages

//   constructor(private http: HttpClient) {
//     this.loginobj = new Loginveb();
//     this.signupobj = new signUpveb();
//     this.errorMessage = ''; // Initialize to empty string
//   }

//   onLogin(): void {

//     const body = new URLSearchParams();
//     body.set('username', this.loginobj.username);
//     body.set('password', this.loginobj.password);

//     let headers = new HttpHeaders();
//     headers = headers.set("accept", "application/json");
//     headers = headers.set('Content-Type', 'application/x-www-form-urlencoded');

//     this.http.post('http://127.0.0.1:8000/login', body.toString(), {headers: headers} ).subscribe({
//       next: (response) => {
//         console.log('Login successful', response);
//         // Handle successful login, e.g., navigate or store the token
//       },
//       error: (error) => {
//         // Set the error message from the API
//         console.error('Login error:', error);
//         this.errorMessage = error.statusText;
//       }
//     });
//   }

//   onSignUp(): void {
//     this.http.post('http://127.0.0.1:8000/users', this.signupobj).subscribe({
//       next: (response) => {
//         console.log('Signup successful', response);
//         // Handle successful signup
//       },
//       error: (error) => {
//         // Set the error message from the API
//         this.errorMessage = error.message || 'An error occurred during signup.';
//         console.error('Signup error:', error);
//       }
//     });
//   }


//   isApproved = false

//   signUp(status: string): void {
//     if (status === "login") {
//       this.isApproved = true;
//     } else if (status === "signup") {
//       this.isApproved = false;
//     }
//   }

//   continueClicked() {

//     console.log("Continue Clicked !!");
//   }
// }


// export class Loginveb {
//   username: string;
//   password: string;
//   constructor () {
//   this.username = "";
//   this.password = "";
//   }
// }

// export class signUpveb {
//   email: string;
//   password: string;
//   first_name: string;
//   last_name: string;
//   constructor () {
//   this.email = "";
//   this.password = "";
//   this.first_name = "";
//   this.last_name = "";
//   }

// }
