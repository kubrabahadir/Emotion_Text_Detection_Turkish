# Emotion Text Detection Turkish 
Student by Kübra Bahadır 
190401014

# Emotion Detection in Turkish Text

This repository contains a project focusing on detecting emotions from text data using machine learning techniques. The project is implemented as a Streamlit web application allowing users to input Turkish text and analyze the emotions expressed in the text.

## Features

- **Emotion Prediction**: The application takes Turkish text as input, translates it to English using the Google Translate API, and then uses a pre-trained machine learning model to predict the emotion expressed in the text.
- **Visualization**: The predicted emotion is displayed along with an appropriate emoji and the prediction probability. The prediction probabilities for all emotions are also visualized using a bar chart.
- **Database Integration**: The application uses an SQLite database to keep track of page visits and prediction details, allowing for easy data tracking and analysis.
- **User-Friendly Interface**: The Streamlit web application provides a simple and intuitive user interface for inputting text and viewing the results.

## Technologies Used

- Python
- Streamlit
- Google Translate API
- SQLite
- scikit-learn
- Pandas
- NumPy
- Altair
- Plotly

## Getting Started

To run the project locally, follow these steps:

1. Clone the repository: `git clone https://github.com/your-username/emotion-detection-text.git`
2. Create a virtual environment (recommended):
   - Windows: `python -m venv myenv` (replace `myenv` with your desired environment name)
   - macOS/Linux: `python3 -m venv myenv`
3. Activate the virtual environment:
   - Windows: `myenv\Scripts\activate`
   - macOS/Linux: `source myenv/bin/activate`
4. Install the required dependencies: `pip install -r requirements.txt`
5. Download the pre-trained model file and place it in the `models` directory.
6. Run the Streamlit app: `streamlit run app2.py`
7. Access the application in your web browser at the provided local URL.

## Contributing

Contributions to this project are welcome. If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## Acknowledgments

The project was inspired by the need for emotion analysis in text data. Special thanks to the developers of the libraries and frameworks used in this project.

