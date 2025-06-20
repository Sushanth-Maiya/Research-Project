import os
import pandas as pd

audio_dir = 'audio'
output_excel = 'horn_intents_updated.xlsx'

intent_map = {
    'L': 'Warning',
    'S': 'Friendly Greeting',
    'V': 'Frustration'
}

direction_map = {
    'A': 'Ahead',
    'B': 'Back',
    'S': 'Side'
}

data = []

for filename in os.listdir(audio_dir):
    if filename.lower().endswith('.wav'):
        name = filename.replace('.wav', '').replace('.WAV', '')
        parts = [p for p in name.split('_') if p]  # remove blanks

        print(f"Processing: {filename} → {parts}")

        horn_presence = 1  # default: horn
        intent = 'Unknown'
        direction = 'Unknown'

        # Detect not-horn
        for part in parts:
            if part.upper().startswith('N'):  # e.g., N1, N9
                horn_presence = 0
                intent = 'None'
                break

        # Detect intent (L, S, V)
        if horn_presence == 1:
            for part in reversed(parts):
                if part.upper() in intent_map:
                    intent = intent_map[part.upper()]
                    break

        # Detect direction (first match: A, B, S)
        for part in parts:
            if part.upper() in direction_map:
                direction = direction_map[part.upper()]
                break

        data.append({
            'filename': filename,
            'horn_presence': horn_presence,
            'intent': intent,
            'direction': direction
        })

# Create DataFrame
df = pd.DataFrame(data)
df.to_excel(output_excel, index=False)

print(f"\n✅ Excel file created: {output_excel}")
print(df.head(10))