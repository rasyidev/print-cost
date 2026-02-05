# Print-Cost

Automated document printing cost estimation using Machine Learning.

## Overview

**Print-Cost** is a Machine Learning project inspired by my personal experience running a small printing business during college.  
Back then, printing costs were calculated manually—slow, inconsistent, and difficult to scale during busy hours.

This project automates the pricing process by analyzing document content and estimating ink usage per page, resulting in a faster and more transparent pricing system.

👉 **Live Demo**: https://rasyidev-print-cost.hf.space  
📖 **Full Project Write-up**: https://rasyidev.pages.dev/projects/print-cost

---

## What It Does

- Automatically estimates printing cost from PDF documents
- Classifies pages based on color intensity and ink usage
- Calculates total price in seconds
- Presents results with **simple, interactive visualizations**:
  - Total pages & final price
  - Pie chart of price categories
  - Tooltips explaining cost distribution

---

## Impact (Why It Matters)

- 🚀 **109× faster** than manual pricing
- ⏱️ **8.13 seconds** to process an 884-page document
- 🎯 **99% F1 Score**
- Designed to be **business-friendly**:
  - Avoids overcharging
  - Improves price transparency for customers

---

## Tech Stack

- **Python**
- **Scikit-learn**, **XGBoost**
- **FastAPI**
- **PDF & Image Processing** (PyMuPDF, pdf2image, pypdfium2)
- **Interactive Data Visualization**

---

## Learn More

This README focuses on the *what* and *why*.  
For a detailed explanation of the data, modeling decisions, and experiments:

👉 https://rasyidev.pages.dev/projects/print-cost

---

**Author**  
Habib Abdurrasyid
