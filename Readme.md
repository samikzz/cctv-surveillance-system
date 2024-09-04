This project consists of training and UI portion of CCTV surveillance system which detects accident, theft and fight from cctv fooatge. For this project, custom dataset is created by clipping numerous cctv footage into 5s clips containing those actions. Every scene consists of at least 100 high quality video clips of cctv footage (You can email me if you want the dataset and the model). A video vision transformer model is used that captures spatial as well as temporal information from the video in order to classify those actions.

It is better to have python 3.9 installed for better compatibility.

To run this project, download the necesaary library, simply start the uvicorn and then run streamlit
```
uvicorn main:app --reload
streamlit run app.py 
```
