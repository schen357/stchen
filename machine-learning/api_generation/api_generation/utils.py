"""
Reusable methods for API calling
"""
MODEL = "claude-sonnet-5"
MAX_TOKENS = 4000

def call_anthropic_api(client, system_prompt, user_message):
    message = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": user_message
                }
            ]
        )
    return message