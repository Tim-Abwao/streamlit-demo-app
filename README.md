# Streamlit Demo Application

The code for the [Streamlit][st] demo app discussed in the [Deploying Machine Learning Models With Streamlit][st-article] article.

[st]: https://streamlit.io/
[st-article]: https://tim-abwao.github.io/2023/07/Deploying-Machine-Learning-Models-with-Streamlit

![screencast of the app](https://github.com/Tim-Abwao/tim-abwao.github.io/blob/main/assets/images/articles/streamlit-demo/screencast.gif?raw=true)

## Running locally

1. Fetch the necessary files:

   ```bash
   git clone https://github.com/Tim-Abwao/streamlit-demo-app.git
   cd streamlit-demo-app
   ```

2. Create a virtual environment, and install dependencies:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -U pip
   pip install -r requirements.txt
   ```

3. Create and save a model:

   ```python
   python modelling.py
   ```

4. Launch the *Streamlit* app:

   ```bash
   streamlit run streamlit_app.py
   ```
