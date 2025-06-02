import os
import json
import sys
from groq import Groq

# Initialize Groq client with API key from environment
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Get the prompt from command-line arguments
prompt = sys.argv[1] if len(sys.argv) > 1 else os.getenv("PROMPT", "")

# Define the model
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
        temperature=1,
        max_completion_tokens=1024,
        top_p=1,
        stream=False,
        stop=None,
        response_format={"type": "json_object"}  # Enforce JSON output
    )

    # Get the raw response
    raw_response = completion.choices[0].message.content
    print("Raw Groq API Response:")
    print(raw_response)
    print("--- End of Raw Response ---")

    # Save raw response to a file for debugging
    with open("raw_response.txt", "w") as f:
        f.write(raw_response)

    # Attempt to parse the response as JSON
    try:
        json_output = json.loads(raw_response)
        required_keys = {"job_title", "job_link", "company", "city", "remote"}
        if not all(key in json_output for key in required_keys):
            print(f"Error: Missing required keys in JSON output. Found keys: {json_output.keys()}")
            sys.exit(1)

        # Write the parsed JSON to a file
        with open("job_output.json", "w") as f:
            json.dump(json_output, f, indent=2)

        # Print parsed JSON for debugging
        print("Parsed JSON Output:")
        print(json.dumps(json_output, indent=2))

    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON response from Groq API: {str(e)}")
        print("Raw response saved to raw_response.txt for inspection")
        sys.exit(1)

except Exception as e:
    print(f"Error calling Groq API: {str(e)}")
    sys.exit(1)
