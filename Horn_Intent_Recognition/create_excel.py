import os
import pandas as pd

audio_dir = 'audio'

label_map = {
    'L': 'Warning about a hazard',
    'S': 'Indicating presence',
    'V': 'Frustration or hazard'
}

data = []

for filename in os.listdir(audio_dir):
    if filename.upper().endswith('.WAV'):
        name_no_ext = filename.replace('.WAV', '').replace('.wav', '')
        parts = name_no_ext.split('_')
        parts = [p for p in parts if p]  # remove empty strings

        print(f"Processing: {filename} → parts: {parts}")  # Debug

        # Try last part first, then second-last
        for part in reversed(parts):
            part = part.upper()
            if part in label_map:
                data.append({
                    'filename': filename,
                    'intent': label_map[part]
                })
                print(f"✅ Intent matched: {label_map[part]}")
                break
        else:
            print(f"❌ No intent label found in: {filename}")

# Save to Excel
df = pd.DataFrame(data)
df.to_excel('horn_intents.xlsx', index=False)
print("\n✅ Excel file created: horn_intents.xlsx")
