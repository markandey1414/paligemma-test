# paligemma-test

Pali Gemma is Vision Language model released by Google. It takes images and text as input, combines their embeddings andproduces the effective result as per the given prompt.
This is an attempt to re-create a local reconstruction of the model using CPU resources.
Original model: https://huggingface.co/google/paligemma-3b-pt-224

create a virtual environment
`python -m venv venv`

activate it
`source venv/Scripts/activate`

install requirements
`pip install -r requirements.txt`

run the 'launch.sh' file
`./launch.sh`
