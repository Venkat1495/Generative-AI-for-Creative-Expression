import { Routes } from '@angular/router';
import { LoginComponent } from './login/login.component'
import { PredictionComponent } from './prediction/prediction.component';

export const routes: Routes = [
    {path:'', redirectTo: 'login', pathMatch: "full"},
    {path: 'login', component: LoginComponent },
    {path: 'prediction', component: PredictionComponent},
    {path: '**', component: LoginComponent}
];
