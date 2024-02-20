from openai import OpenAI
import argparse
import httpx
import ast
import re

def update_template(prompt, template_file):
    with open(template_file, 'r') as file:
        lines = file.readlines()
    index = len(lines) - 3

    lines[index] = f"Caption: {prompt}\n"

    updated_template = ''.join(lines)
    return updated_template



def get_layout(user_prompt, api_key):
    client = OpenAI(
        base_url="xxx", 
        api_key=api_key,
        http_client=httpx.Client(
            base_url="xxx",
            follow_redirects=True,
        ),
    )
    prompt = update_template(user_prompt, "template.txt")

    completion = client.chat.completions.create(
      model="gpt-4",
      messages=[
        {"role": "system", "content": prompt},
        {"role": "user", "content": prompt}
      ]
    )

    output = completion.choices[0].message.content
    output = str(output)

    result = re.search(r"\[(.*?)\]\nPosition:\s*(\[(.*?)\])", output)
    extracted_part = result.group(1)
    obj = ast.literal_eval(extracted_part)
    phrase = [item[0] for item in obj]
    location = [item[1] for item in obj]
    token_location = ast.literal_eval(result.group(2))
    
    return phrase, location, token_location


