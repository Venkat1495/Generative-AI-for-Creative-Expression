Step-by-Step Guide to Start the Project
---------------------------------------

### Create a Local Virtual Environment

For Windows:

-   Open Command Prompt and navigate to your project directory.

-   Create the virtual environment by running 'python -m venv myenv'.

-   Activate the virtual environment with 'myenv\Scripts\activate'.

For macOS and Linux:

-   Open Terminal and navigate to your project directory.

-   Create the virtual environment with 'python3 -m venv myenv'.

-   Activate the virtual environment with 'source myenv/bin/activate'.

### Install Required Libraries

Install all dependencies listed in your requirements.txt file by executing pip install -r requirements.txt.

### Prepare the Dataset

Assuming you have your raw data, use the data_format.py script to process and split the data into train.bin and test.bin files:

-   Run the data formatting script with python data_format.py.

### Train the Model

Modify the config.py file according to your requirements to adjust hyperparameters. Then, initiate the training process by running python train_with.py. Monitor the output to ensure the model is training and the losses are decreasing as expected.

### Set Up Backend Configuration

Ensure your MongoDB credentials, OpenAI API key, and SECRET_KEY are correctly set in BackEnd/Core/security.py and BackEnd/Core/database.py:

-   Review and update these configurations as necessary to ensure they are properly configured and secured.

### Launch the Backend Server

Start the backend server using FastAPI which will serve as the API backend for your frontend application:

-   Navigate to the directory containing main.py.

-   Start the FastAPI application using uvicorn main:app --reload. The --reload option makes the server restart after code changes, which is useful during development.

### Start the Angular Frontend

Navigate to the frontend directory and start the Angular application:

-   Change directory to the frontend application.

-   Install npm packages with npm install and start the Angular application with npm start.This will compile the Angular application and typically serve it on http://localhost:4200.

### Testing the Application

Thoroughly test the application by navigating through the frontend and ensuring that all functionalities such as user registration, login, prediction generation, and history retrieval are working as expected.