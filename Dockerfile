FROM python:3.6.5-slim
COPY pipeline/scripts/labeller.py labeller.py
COPY pipeline/scripts/utils.py utils.py
COPY final_model final_model
COPY requirements-lite.txt requirements-lite.txt
RUN pip3 install -r requirements-lite.txt
CMD ["python3", "labeller.py", "--host=0.0.0.0"]
