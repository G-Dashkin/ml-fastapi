from pathlib import Path
import pandas as pd

def load_dataset(): return pd.read_csv(Path(__file__).parent.parent.parent / "data" / "churn_dataset.csv")