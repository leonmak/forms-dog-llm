# forms-dog-llm

## Install python packages

- `pip install langchain transformers torch flask`
- `pip install -r requirements.txt`

## Add local data

- Add a text file containing examples of question answer pair to be used in apply form
- They will be used in the context part of the prompt to the LLM

Example file in `./data/job_application.txt`:

```
Q: First Name
A: L

Q: Last Name
A: M

Q: Job title
A: Software Engineer

Q: Country
A: Singapore

Q: Email Address
A: abc@gmail.com

Q: Phone Number
A: 123123

Q: Please list 2-3 dates and time ranges for interview
A: Weekdays 9am - 12pm, 3pm - 5pm

Q: LinkedIn URL
A: https://www.linkedin.com/in/lm

Q: Most recent Work Experience
A: Role: Software Engineer
Company: Acme
Date: Dec 2020 to March 2023
```

## Run local server

- Run flask locally `python app.py`
- Note the endpoint: `127.0.0.1:8000` which is called from chrome extension
