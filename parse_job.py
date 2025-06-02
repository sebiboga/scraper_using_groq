import os
import json
from groq import Groq

# Initialize Groq client with API key from environment
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Get the prompt from command-line arguments or GitHub Actions input
import sys
prompt = sys.argv[1] if len(sys.argv) > 1 else os.getenv("GITHUB_EVENT_INPUTS_PROMPT", "")

# Define the model (can be overridden in the YAML if needed)
MODEL = "compound-beta"

# Define the system prompt to ensure structured JSON output
system_prompt = """
You are a job listing parser. Given a job description or prompt, extract and return a JSON object with the following keys: job_title, job_link, company, city, and remote. If any information is missing, use null for that field. Ensure the output is valid JSON.

Example output:
{
  "job_title": "Software Engineer",
  "job_link": "https://example.com/jobs/123",
  "company": "Example Corp",
  "city": "San Francisco",
  "country": "România",
  "remote": true
}
"""

try:
    # Create the chat completion
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5,
        max_completion_tokens=2048,
        top_p=1,
        stream=False,  # Stream set to False for simplicity
        stop=None,
        response_format={"type": "json_object"}  # Enforce JSON output
    )

    # Extract the JSON response
    result = completion.choices[0].message.content

    # Parse and validate the JSON
    try:
        json_output = json.loads(result)
        required_keys = {"job_title", "job_link", "company", "city", "remote"}
        if not all(key in json_output for key in required_keys):
            raise ValueError("Missing required keys in JSON output")
        
        # Write the output to a file
        with open("job_output.json", "w") as f:
            json.dump(json_output, f, indent=2)
        
        # Print for debugging
        print(json.dumps(json_output, indent=2))

    except json.JSONDecodeError:
        print("Error: Invalid JSON response from Groq API")
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

except Exception as e:
    print(f"Error calling Groq API: {str(e)}")
    sys.exit(1)
