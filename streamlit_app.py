import joblib
import streamlit as st

st.title("Iris Species Classifier")
st.write(
    "Predict the iris flower species from petal and sepal dimensions. ",
    "Adjust the dimensions to get a prediction.",
)


@st.cache_resource
def load_model():
    """Fetch and cache the saved model: avoids reloading it with each rerun/
    event.

    Returns:
        RandomForestClassifier: Trained Scikit-learn model.
    """
    return joblib.load("model.gz")


model = load_model()
species_dict = {0: "setosa", 1: "versicolor", 2: "virginica"}
image_attributions = dict(
    setosa="Денис Анисимов, Public domain, via Wikimedia Commons",
    versicolor="D. Gordon E. Robertson, CC BY-SA 3.0, via Wikimedia Commons",
    virginica="Eric Hunt, CC BY-SA 4.0, via Wikimedia Commons",
)
# Set a 2-column layout: Dimensions | Prediction
col1, col2 = st.columns([0.35, 0.65], gap="medium")
with col1:
    st.subheader("Dimensions")
    input_data = [
        st.number_input(
            dim, max_value=10.0, min_value=0.0, step=0.1, value=5.0, format="%.1f"
        )
        for dim in [
            "Sepal length (cm)",
            "Sepal width (cm)",
            "Petal length (cm)",
            "Petal width (cm)",
        ]
    ]
    st.write(f"Input data:\n :blue[{[round(x, 1) for x in input_data]}]")

with col2:
    st.subheader("Prediction")
    pred = model.predict([input_data])  # array of length 1 e.g [2]
    predicted_species = species_dict.get(pred[0])
    st.write(f"Species: :green[Iris {predicted_species}]")
    st.image(
        f"assets/iris-{predicted_species}.jpg",
        width=320,
        caption=f"Source: {image_attributions[predicted_species]}",
    )
