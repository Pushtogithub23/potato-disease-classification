import gradio as gr
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io

# Load the trained model
MODEL_PATH = "SAVED_MODELS/model_1.keras"  # Update this path to your saved model
model = tf.keras.models.load_model(MODEL_PATH)

# Define class names (should match your training data)
CLASS_NAMES = ["Potato___Early_blight", "Potato___Late_blight", "Potato___healthy"]

# Image preprocessing parameters
IMAGE_SIZE = 256


def preprocess_image(image):
    """
    Preprocess the input image for model prediction
    """
    # Convert PIL image to numpy array
    if isinstance(image, Image.Image):
        image = np.array(image)

    # Add batch dimension
    image = tf.expand_dims(image, axis=0)

    return image


def create_probability_plot(probabilities, class_names):
    """
    Create a bar plot of prediction probabilities
    """
    # Set dark theme
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.subplots_adjust(top=0.9, bottom=0.2)
    fig.patch.set_facecolor("#1e1e1e")
    ax.set_facecolor("#1e1e1e")

    # Create color map - highlight the highest probability with modern colors
    colors = [
        "#ff6b6b" if i != np.argmax(probabilities) else "#4ecdc4"
        for i in range(len(probabilities))
    ]

    bars = ax.bar(
        class_names,
        probabilities * 100,
        color=colors,
        edgecolor="white",
        linewidth=1.5,
        alpha=0.9,
    )

    # Add percentage labels on bars with white text
    for bar, prob in zip(bars, probabilities):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 1,
            f"{prob * 100:.1f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
            color="white",
            fontsize=11,
        )

    # Style the plot with white text
    ax.set_title(
        "Potato Disease Classification Probabilities",
        fontsize=16,
        fontweight="bold",
        color="white",
        pad=20,
    )
    ax.set_xlabel("Disease Classes", fontsize=14, color="white")
    ax.set_ylabel("Probability (%)", fontsize=14, color="white")
    ax.set_ylim(0, 105)

    # Style tick labels
    ax.tick_params(axis="x", colors="white", labelsize=12)
    ax.tick_params(axis="y", colors="white", labelsize=12)

    # Add subtle grid for better readability
    ax.grid(axis="y", alpha=0.3, color="gray", linestyle="--")

    # Remove top and right spines for cleaner look
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("white")
    ax.spines["bottom"].set_color("white")

    plt.tight_layout()

    # Convert plot to image
    buf = io.BytesIO()
    plt.savefig(
        buf,
        format="png",
        dpi=150,
        bbox_inches="tight",
        facecolor="#1e1e1e",
        edgecolor="none",
    )
    buf.seek(0)
    plt.close()

    # Reset to default style to avoid affecting other plots
    plt.style.use("default")

    return Image.open(buf)


def get_disease_info(predicted_class):
    """
    Return information about the predicted disease
    """
    disease_info = {
        "Potato___Early_blight": {
            "description": "Early blight is a common potato disease caused by Alternaria solani. It appears as dark spots with concentric rings on leaves.",
            "treatment": "Use fungicides, practice crop rotation, and ensure proper plant spacing for air circulation.",
            "severity": "Moderate",
        },
        "Potato___Late_blight": {
            "description": "Late blight is a serious potato disease caused by Phytophthora infestans. It can cause rapid destruction of leaves and tubers.",
            "treatment": "Apply fungicides preventively, remove infected plants, and avoid overhead watering.",
            "severity": "High",
        },
        "Potato___healthy": {
            "description": "The potato plant appears healthy with no visible signs of disease.",
            "treatment": "Continue regular monitoring and maintain good agricultural practices.",
            "severity": "None",
        },
    }

    return disease_info.get(
        predicted_class,
        {
            "description": "Unknown disease classification.",
            "treatment": "Consult with agricultural experts.",
            "severity": "Unknown",
        },
    )


def predict_disease(image):
    """
    Main prediction function for Gradio interface
    """
    if image is None:
        return "Please upload an image", None, ""

    try:
        # Preprocess the image
        processed_image = preprocess_image(image)

        # Make prediction
        predictions = model.predict(processed_image, verbose=0)
        probabilities = predictions[0]

        # Get predicted class
        predicted_class_idx = np.argmax(probabilities)
        predicted_class = CLASS_NAMES[predicted_class_idx]
        confidence = probabilities[predicted_class_idx] * 100

        # Create probability plot
        disease_names = [
            name.replace("Potato__", "").replace("_", " ").title()
            for name in CLASS_NAMES
        ]
        plot_image = create_probability_plot(probabilities, disease_names)

        # Get disease information
        disease_info = get_disease_info(predicted_class)

        # Format results
        result_text = f"""
                    ## 🔍 **Prediction Results**

                    **Predicted Disease:** {predicted_class.replace("Potato___", "").replace("_", " ").title()}
                    **Confidence:** {confidence:.2f}%

                    ## 📋 **Disease Information**

                    **Description:** {disease_info["description"]}

                    **Recommended Treatment:** {disease_info["treatment"]}

                    **Severity Level:** {disease_info["severity"]}

                    ---
                    *Note: This is an AI-based prediction. For critical decisions, please consult with agricultural experts.*
                    """

        return (
            result_text,
            plot_image,
            f"Prediction: {predicted_class.replace('Potato___', '').replace('_', ' ').title()} ({confidence:.1f}%)",
        )

    except Exception as e:
        return f"Error processing image: {str(e)}", None, "Error occurred"


# Create Gradio interface
def create_gradio_app():
    """
    Create and configure the Gradio interface
    """

    # Custom CSS for better styling
    css = """
    .gradio-container {
        font-family: 'Arial', sans-serif;
    }
    .output-markdown {
        font-size: 14px;
    }
    .image-upload {
        border: 2px dashed #4CAF50;
        border-radius: 10px;
    }
    """

    with gr.Blocks(css=css, title="Potato Disease Classifier") as app:
        gr.Markdown("""
        # 🥔 Potato Disease Classification System

        Upload an image of a potato leaf to detect potential diseases. The system can identify:
        - **Early Blight** - Caused by Alternaria solani
        - **Late Blight** - Caused by Phytophthora infestans
        - **Healthy** - No disease detected

        ---
        """)

        with gr.Row():
            with gr.Column(scale=1):
                # Input section
                gr.Markdown("### 📤 Upload Image")
                input_image = gr.Image(
                    type="pil",
                    label="Upload a potato leaf image",
                    elem_classes=["image-upload"],
                )

                predict_btn = gr.Button(
                    "🔍 Analyze Disease", variant="primary", size="lg"
                )

                # Quick status
                status_text = gr.Textbox(label="Status", interactive=False, max_lines=1)

            with gr.Column(scale=2):
                # Output section
                gr.Markdown("### 📊 Results")

                with gr.Row():
                    result_text = gr.Markdown(
                        value="Upload an image and click 'Analyze Disease' to see results.",
                        elem_classes=["output-markdown"],
                    )

                probability_plot = gr.Image(
                    label="Probability Distribution", type="pil"
                )

        # Examples section
        gr.Markdown("### 🖼️ Example Images")
        gr.Markdown("*Click on any example image below to test the classifier:*")

        # You can add example images here if you have them
        gr.Examples(
            examples=[
                ["TEST_IMAGES/Early_Blight.JPG"],
                ["TEST_IMAGES/Late_Blight.JPG"],
                ["TEST_IMAGES/Healthy.JPG"],
            ],
            inputs=input_image,
        )

        # Connect the prediction function
        predict_btn.click(
            fn=predict_disease,
            inputs=[input_image],
            outputs=[result_text, probability_plot, status_text],
        )

        # Also allow prediction on image upload
        input_image.change(
            fn=predict_disease,
            inputs=[input_image],
            outputs=[result_text, probability_plot, status_text],
        )

        gr.Markdown("""
        ---
        ### ℹ️ **Important Notes:**
        - Ensure the image clearly shows potato leaves
        - Good lighting and focus improve accuracy
        - This tool is for educational/research purposes
        - Always verify results with agricultural experts for critical decisions

        **Model Accuracy:** ~98.5% on test dataset
        """)

    return app


# Launch the app
if __name__ == "__main__":
    try:
        # Create and launch the Gradio app
        app = create_gradio_app()

        # Launch with custom settings
        app.launch(
            share=True,  # Set to True to create a public link
            server_name="0.0.0.0",  # Allow access from any IP
            server_port=7860,  # Default Gradio port
            debug=True,  # Enable debug mode
            show_error=True,  # Show detailed error messages
        )

    except Exception as e:
        print(f"Error launching Gradio app: {e}")
        print("Make sure you have installed all required packages:")
        print("pip install gradio tensorflow pillow matplotlib seaborn")
