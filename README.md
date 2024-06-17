# Disaster Response Pipeline Project

## Table of Contents
1. [Introduction](#introduction)
2. [Project Motivation](#project-motivation)
3. [File Descriptions](#file-descriptions)
4. [Installation](#installation)
5. [Instructions](#instructions)
6. [Results](#results)
7. [Licensing, Authors, and Acknowledgements](#licensing-authors-and-acknowledgements)

## Introduction
The Disaster Response Pipeline Project is part of the Udacity Data Scientist Nanodegree. This project aims to build a machine learning pipeline to categorize real messages from disaster events. These messages can come from social media, news articles, and other sources. The categorized messages can then be sent to the appropriate disaster response agencies.

## Project Motivation
During disaster events, timely and accurate information dissemination is crucial for effective response and relief operations. Automating the categorization of messages can help ensure that critical information reaches the appropriate agencies promptly, thereby improving the overall response efforts.

## File Descriptions
The repository contains the following files and directories:

1. **app**: This directory contains the Flask web application files.
   - `run.py`: Flask application entry point.
   - `templates/`: HTML templates for the web application.
   
2. **data**: This directory contains data processing files.
   - `disaster_messages.csv`: The dataset containing disaster-related messages.
   - `disaster_categories.csv`: The dataset containing categories for the messages.
   - `data` : https://www.kaggle.com/datasets/hoangziet/data-augmented-udacity-disaster-response/settings
   - `process_data.ipynb`: Script to load, clean, and save data to a SQLite database.
   - `DisasterResponse.db`: SQLite database containing the cleaned data.
   
3. **models**: This directory contains machine learning pipeline files.
   - `train_classifier.py`: Script to train and save the machine learning model.
   - `best_model1.h5` :https://drive.google.com/file/d/15xfjyaWdx8mzuCHZXH4uh5E6Gy02VFnE/view?usp=sharing
   - `best_model2.h5` :https://drive.google.com/file/d/16o5kOoiqk--ppYPhje3rk5rnkFyAGR7y/view?usp=sharing
   - `best_model3.h5` : https://drive.google.com/file/d/1TOtQiuGh-tpQlqdR87YztuDHjiereyLZ/view?usp=sharing
   

4. **README.md**: Project documentation.

## Installation
To run this project, you will need to have the following dependencies installed:
- Python 3.x
- pandas
- numpy
- scikit-learn
- sqlalchemy
- tensorflow
- Flask
- plotly

You can install these dependencies using pip:
```bash
pip install pandas numpy scikit-learn sqlalchemy nltk Flask plotly
```

## Instructions
1. **Run process_data**: Process data and save to database.
   ```bash
   python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
   ```

2. **Run ML Pipeline**: Train the model and save it.
   ```bash
   python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
   ```

3. **Run the Web App**:
   ```bash
   python app/run.py
   ```

4. **Interact with the Web App**: Go to http://localhost:3001/ to view the app and classify disaster messages.

## Results
The machine learning model categorizes disaster messages into one or more of 36 categories, including 'related', 'request', 'offer', 'aid_related', 'medical_help', 'search_and_rescue', etc. The web application allows users to input new messages and get real-time classification results.

## Licensing, Authors, and Acknowledgements
This project is part of the Udacity Data Scientist Nanodegree program. The data comes from Figure Eight. Special thanks to Udacity for providing the project template and Figure Eight for the datasets. 

If you have any questions or suggestions, feel free to reach out to the author.

---

Thank you for using the Disaster Response Pipeline Project!
