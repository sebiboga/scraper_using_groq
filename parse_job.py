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

# Estimate token count (rough: 1 token ≈ 4 characters)
def estimate_tokens(text):
    return len(text) // 4 + 1

# Define a conservative max token limit (adjust based on model)
MAX_TOKENS = 4096  # Conservative limit; adjust if model supports more

# Define a concise system prompt
system_prompt = """
Parse the job description into a JSON object with keys: job_title, job_link, company, city, remote. Use null for missing fields. Return only valid JSON, no extra text.

Example:
{
  "job_title": "Software Engineer",
  "job_link": "https://example.com/jobs/123",
  "company": "Example Corp",
  "city": "San Francisco",
  "remote": true
}
"""

# Log prompt details for debugging
total_prompt = system_prompt + prompt
print(f"Input Prompt Length: {len(prompt)} characters, ~{estimate_tokens(prompt)} tokens")
print(f"Total Request Length (with system prompt): {len(total_prompt)} characters, ~{estimate_tokens(total_prompt)} tokens")
print("Input Prompt Content:")
print(prompt)
print("--- End of Input Prompt ---")

# Check if prompt is too large
if estimate_tokens(total_prompt) > MAX_TOKENS:
    print(f"Error: Prompt too large (~{estimate_tokens(total_prompt)} tokens exceeds {MAX_TOKENS} token limit). Please shorten the prompt.")
    # Optionally trim the prompt (uncomment to enable)
    # prompt = prompt[:MAX_TOKENS * 4 - len(system_prompt)]
    # print("Prompt trimmed to fit token limit.")
    sys.exit(1)

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
