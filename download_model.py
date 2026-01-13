import os
from huggingface_hub import hf_hub_download

def download_llm():
    # Korrigierte Quelle für Lexi-Uncensored
    model_name = "mradermacher/Llama-3.1-8B-Lexi-Uncensored-GGUF"
    # Wir nehmen die Q5_K_M Version für die beste Balance aus Speed und Intelligenz
    filename = "Llama-3.1-8B-Lexi-Uncensored.Q5_K_M.gguf"
    folder = "models"
    
    if not os.path.exists(folder):
        os.makedirs(folder)
        
    print(f"Downloading {filename} from {model_name}...")
    
    try:
        path = hf_hub_download(
            repo_id=model_name,
            filename=filename,
            local_dir=folder
        )
        print(f"\nErfolg! Modell gespeichert unter: {path}")
    except Exception as e:
        print(f"\nFehler beim Download: {e}")
        print("Hinweis: Falls der Fehler '401' bleibt, stelle sicher, dass du eingeloggt bist oder das Repo öffentlich ist.")

if __name__ == "__main__":
    download_llm()