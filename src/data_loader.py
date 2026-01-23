import pandas as pd 
#from src.utils import print_header


def load_dataset(path: str) -> pd.DataFrame:
    """Load dataset
    
        Args: path
    """

    #print_header("Loading dataset")
    df = pd.read_csv(path)
    print(f'Dataset is loaded with shape: {df.shape}')
    return df 


if __name__ == "__main__":
    df = load_dataset('data/phising.csv')
    print(df.sample())

