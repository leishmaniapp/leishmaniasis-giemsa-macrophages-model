#! Requires model_wrapper repository
# (visit for more information https://github.com/leishmaniapp/model_wrapper)
FROM leishmaniapp/model-wrapper AS wrapper

# Use the python interpreter
FROM python:slim

WORKDIR /app

# Copy source files
COPY . .

# Download dependencies
RUN <<EOF
# System dependencies
apt update
apt install -y libgl1 libglib2.0-0
# Python dependencies
python -m ensurepip --upgrade
pip install -r requirements.txt
EOF

# Copy the model wrapper
COPY --from=wrapper /root/model_wrapper .

CMD ["./model_wrapper"]