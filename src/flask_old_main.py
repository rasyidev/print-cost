from flask import Flask, render_template, request, redirect, url_for
from helper import predict_pdf, label_to_price


app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "GET":
        return render_template("index.html")
    elif request.method == "POST":
        f = request.files["pdf_file"]

        output_df = predict_pdf(f)
        result = label_to_price(output_df)
        return render_template("index.html", data=result)


@app.route("/calculate_print_cost", methods=["POST"])
def calculate_print_cost():
    f = request.files["pdf_file"]
    output_df = predict_pdf(f)
    result = label_to_price(output_df)
    return redirect(url_for("index"))


from pathlib import Path
import sys, os

PROJECT_CONFIG = Path(__file__).resolve().parent.parent
sys.path.append(os.path.abspath(__file__))
from config import PROJECT_ROOT

print(PROJECT_ROOT)


if __name__ == "__main__":
    app.run(debug=True)
