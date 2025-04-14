import os
import pandas as pd
from sklearn.model_selection import train_test_split
import requests
import zipfile
import io
import csv
import glob

def download_hwu64():
    """
    Download and prepare the HWU64 dataset for continual learning.
    Organize data by domain to enable sequential learning.
    """
    print("Downloading HWU64 dataset from alternative source...")
    
    # Direct download link to the dataset
    download_url = "https://github.com/xliuhw/NLU-Evaluation-Data/archive/master.zip"
    
    try:
        # Download the dataset
        print("Downloading from GitHub repository...")
        response = requests.get(download_url)
        if response.status_code != 200:
            raise Exception(f"Failed to download: status code {response.status_code}")
        
        # Extract the zip file
        print("Extracting dataset...")
        z = zipfile.ZipFile(io.BytesIO(response.content))
        extract_dir = "data/raw"
        os.makedirs(extract_dir, exist_ok=True)
        z.extractall(extract_dir)
        
        print("Dataset downloaded and extracted successfully!")
        
        # Process the dataset
        process_hwu64_data(extract_dir)
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("Please check your internet connection and try again.")
        return

def process_hwu64_data(raw_data_dir):
    """Process the raw HWU64 dataset files."""
    print("Processing HWU64 dataset...")
    
    # Look for CSV-like files in the AnnotatedData directory
    base_dir = os.path.join(raw_data_dir, "NLU-Evaluation-Data-master")
    annotated_dir = os.path.join(base_dir, "AnnotatedData")
    
    # If AnnotatedData directory doesn't exist, search in the whole repository
    if not os.path.exists(annotated_dir):
        print(f"AnnotatedData directory not found, searching the entire repository...")
        annotated_dir = base_dir
    
    # Look for CSV, TSV, or TXT files
    data_files = []
    for ext in [".csv", ".tsv", ".txt"]:
        data_files.extend(glob.glob(os.path.join(annotated_dir, "**/*" + ext), recursive=True))
    
    print(f"Found {len(data_files)} potential data files")
    
    # Store all valid examples
    all_examples = []
    
    # Try to process each file
    for file_path in data_files[:20]:  # Limit to first 20 for initial testing
        print(f"Checking file: {file_path}")
        
        try:
            # Read a small part of the file to determine its format
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                first_chunk = f.read(1000)
                
                # Check if file might contain our data
                has_scenario = 'scenario' in first_chunk.lower()
                has_intent = 'intent' in first_chunk.lower()
                has_answer = 'answer' in first_chunk.lower()
                
                if has_scenario and has_intent and has_answer:
                    print(f"Found potential HWU64 data file: {file_path}")
                    
                    # Determine the delimiter (semicolon is most likely)
                    delimiter = ';' if ';' in first_chunk else ('\t' if '\t' in first_chunk else ',')
                    
                    # Process the file
                    try:
                        # Try reading as CSV with the identified delimiter
                        df = pd.read_csv(file_path, delimiter=delimiter, encoding='utf-8', 
                                        error_bad_lines=False, warn_bad_lines=True, quoting=csv.QUOTE_NONE)
                        
                        # Check if the dataframe has the columns we need
                        required_cols = ['scenario', 'intent', 'answer_annotation']
                        has_cols = all(col.lower() in df.columns.str.lower() for col in required_cols)
                        
                        if has_cols:
                            print(f"Processing data from file: {file_path}")
                            
                            # Normalize column names to lowercase
                            df.columns = df.columns.str.lower()
                            
                            # Use answer_annotation if available, otherwise use answer
                            text_col = 'answer_annotation' if 'answer_annotation' in df.columns else 'answer'
                            
                            # Extract examples
                            for _, row in df.iterrows():
                                scenario = row['scenario']
                                intent = row['intent']
                                text = row[text_col]
                                
                                # Skip rows with missing values
                                if pd.isna(scenario) or pd.isna(intent) or pd.isna(text):
                                    continue
                                    
                                # Skip irrelevant examples
                                status = row.get('status', '')
                                if isinstance(status, str) and status.upper() == 'IRR':
                                    continue
                                
                                # Clean the text by removing entity annotations
                                import re
                                clean_text = re.sub(r'\[.*?\]', '', text)
                                clean_text = ' '.join(clean_text.split())
                                
                                all_examples.append({
                                    'scenario': scenario.strip(),
                                    'intent': intent.strip(),
                                    'text': clean_text.strip()
                                })
                            
                            print(f"Extracted {len(all_examples)} examples so far")
                    
                    except Exception as e:
                        print(f"Error processing CSV: {e}")
                        
                        # Try reading line by line if CSV parsing fails
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            lines = f.readlines()
                            
                            for line in lines[1:]:  # Skip header
                                parts = line.split(delimiter)
                                if len(parts) >= 5:  # Assuming at least 5 columns
                                    try:
                                        scenario = parts[2].strip().strip('"')
                                        intent = parts[3].strip().strip('"')
                                        text = parts[5].strip().strip('"')  # answer_annotation
                                        
                                        if scenario and intent and text:
                                            # Clean the text
                                            import re
                                            clean_text = re.sub(r'\[.*?\]', '', text)
                                            clean_text = ' '.join(clean_text.split())
                                            
                                            all_examples.append({
                                                'scenario': scenario,
                                                'intent': intent,
                                                'text': clean_text
                                            })
                                    except:
                                        continue
        
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
    
    print(f"Collected {len(all_examples)} examples total")
    
    if not all_examples:
        print("No valid examples found. Creating simulated data instead...")
        create_simulated_data()
        return
    
    # Group by scenario (domain)
    domain_data = {}
    for example in all_examples:
        domain = example['scenario']
        if domain not in domain_data:
            domain_data[domain] = []
        
        domain_data[domain].append({
            'text': example['text'],
            'intent': example['intent']
        })
    
    # Print available domains
    available_domains = list(domain_data.keys())
    print(f"Found {len(available_domains)} domains: {available_domains}")
    
    # Select domains with enough examples (at least 50)
    valid_domains = [domain for domain, examples in domain_data.items() 
                    if len(examples) >= 50]
    
    print(f"Domains with sufficient examples: {valid_domains}")
    
    if not valid_domains:
        print("No domains with sufficient examples. Creating simulated data...")
        create_simulated_data()
        return
    
    # Select 4 domains with the most examples
    domain_sizes = [(domain, len(domain_data[domain])) for domain in valid_domains]
    domain_sizes.sort(key=lambda x: x[1], reverse=True)
    
    selected_domains = [domain for domain, _ in domain_sizes[:4]]
    print(f"Selected domains: {selected_domains}")
    
    # Process each selected domain
    processed_dir = "data/processed"
    os.makedirs(processed_dir, exist_ok=True)
    
    for domain in selected_domains:
        print(f"\nProcessing domain: {domain}")
        examples = domain_data[domain]
        
        # Extract texts and intents
        texts = [ex['text'] for ex in examples]
        intents = [ex['intent'] for ex in examples]
        
        # Get unique intents
        unique_intents = sorted(set(intents))
        print(f"  Number of examples: {len(texts)}")
        print(f"  Number of unique intents: {len(unique_intents)}")
        
        # Create intent mapping
        intent_to_idx = {intent: idx for idx, intent in enumerate(unique_intents)}
        
        # Convert intents to indices
        label_indices = [intent_to_idx[intent] for intent in intents]
        
        # Create DataFrame
        df = pd.DataFrame({
            'text': texts,
            'label': label_indices,
            'original_intent': intents
        })
        
        # Split into train/val/test (70/15/15)
        train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
        
        # Save processed data
        output_path = os.path.join(processed_dir, domain)
        os.makedirs(output_path, exist_ok=True)
        
        train_df.to_csv(os.path.join(output_path, "train.csv"), index=False)
        val_df.to_csv(os.path.join(output_path, "val.csv"), index=False)
        test_df.to_csv(os.path.join(output_path, "test.csv"), index=False)
        
        # Save intent mapping
        with open(os.path.join(output_path, "intent_mapping.txt"), 'w') as f:
            for intent, idx in intent_to_idx.items():
                f.write(f"{idx},{intent}\n")
        
        print(f"  Saved {len(train_df)} train, {len(val_df)} val, {len(test_df)} test examples")
        
        # Print a few examples
        print("\n  Example commands from this domain:")
        for i, row in train_df.head(3).iterrows():
            print(f"    Text: '{row['text']}', Intent: '{row['original_intent']}'")
    
    # Save the list of domains
    with open(os.path.join(processed_dir, "domains.txt"), 'w') as f:
        for domain in selected_domains:
            f.write(f"{domain}\n")
    
    print("\nData preparation complete!")

def create_simulated_data():
    """
    Create simulated data based on the HWU64 dataset domains and intents.
    This is a fallback when the actual dataset can't be processed.
    """
    print("Creating simulated HWU64-like data...")
    
    # Path to save the processed data
    processed_dir = 'data/processed'
    os.makedirs(processed_dir, exist_ok=True)
    
    # Define domains and intents based on the actual HWU64 dataset
    domains_data = {
        'alarm': {
            'intents': ['set', 'remove', 'query'],
            'examples': [
                ["wake me up at five am", "set an alarm for two hours from now", "set alarm at 7am", "wake me up tomorrow morning", "alarm for 6:30pm"],
                ["remove my alarm for tomorrow", "cancel tomorrow's alarm", "delete all alarms", "remove my wake-up call", "stop morning alarm"],
                ["check my alarms", "what alarms do I have set", "is my alarm on", "when is my next alarm", "check morning alarm"]
            ]
        },
        'audio': {
            'intents': ['volume_up', 'volume_down', 'volume_mute'],
            'examples': [
                ["turn up the volume", "make it louder", "increase the volume", "volume up", "louder please"],
                ["turn down the volume", "make it quieter", "decrease the volume", "volume down", "quieter please"],
                ["mute the speakers", "quiet", "no sound", "mute audio", "silence please"]
            ]
        },
        'iot': {
            'intents': ['hue_lightchange', 'hue_lightoff', 'hue_lighton', 'cleaning'],
            'examples': [
                ["change the light to blue", "make the lights pink", "set lights to reading mode", "change light color", "make the lighting warm"],
                ["turn the lights off", "lights off please", "switch off the lights", "turn off all lights", "shut down lights"],
                ["turn on the lights", "lights on please", "switch on the lights", "illuminate the room", "I need some light"],
                ["start vacuum cleaner", "clean the house", "vacuum the floor", "start robot vacuum", "clean the carpet"]
            ]
        },
        'calendar': {
            'intents': ['set', 'query', 'delete'],
            'examples': [
                ["add meeting to calendar", "schedule appointment tomorrow", "create new event", "add reminder", "put in my calendar"],
                ["check my schedule", "what's on my calendar", "any meetings today", "show my appointments", "when is my next event"],
                ["cancel the meeting", "delete the appointment", "remove calendar event", "delete tomorrow's meeting", "clear my schedule"]
            ]
        }
    }
    
    # Process each domain
    selected_domains = list(domains_data.keys())
    
    for domain in selected_domains:
        print(f"\nProcessing domain: {domain}")
        domain_info = domains_data[domain]
        
        # Collect all texts and labels
        all_texts = []
        all_labels = []
        all_intents = []
        
        for idx, intent in enumerate(domain_info['intents']):
            examples = domain_info['examples'][idx]
            all_texts.extend(examples)
            all_labels.extend([idx] * len(examples))
            all_intents.extend([intent] * len(examples))
        
        # Create a dataframe
        df = pd.DataFrame({
            'text': all_texts,
            'label': all_labels,
            'original_intent': all_intents
        })
        
        # Split into train/val/test
        train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
        
        # Save processed data
        output_path = os.path.join(processed_dir, domain)
        os.makedirs(output_path, exist_ok=True)
        
        train_df.to_csv(os.path.join(output_path, "train.csv"), index=False)
        val_df.to_csv(os.path.join(output_path, "val.csv"), index=False)
        test_df.to_csv(os.path.join(output_path, "test.csv"), index=False)
        
        # Save intent mapping
        intent_to_idx = {intent: idx for idx, intent in enumerate(domain_info['intents'])}
        with open(os.path.join(output_path, "intent_mapping.txt"), 'w') as f:
            for intent, idx in intent_to_idx.items():
                f.write(f"{idx},{intent}\n")
        
        print(f"  Saved {len(train_df)} train, {len(val_df)} val, {len(test_df)} test examples")
        
        # Print a few examples
        print("\n  Example commands from this domain:")
        for i, row in train_df.head(3).iterrows():
            print(f"    Text: '{row['text']}', Intent: '{row['original_intent']}'")
    
    # Save the list of domains
    with open(os.path.join(processed_dir, "domains.txt"), 'w') as f:
        for domain in selected_domains:
            f.write(f"{domain}\n")
    
    print("\nSimulated data preparation complete!")

if __name__ == "__main__":
    download_hwu64()