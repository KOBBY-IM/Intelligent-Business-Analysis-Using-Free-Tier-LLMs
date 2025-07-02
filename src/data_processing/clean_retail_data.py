import pandas as pd
import numpy as np

def load_and_analyze_data(filepath: str) -> pd.DataFrame:
    """Load and analyze the shopping trends dataset"""
    
    print("Loading shopping trends dataset...")
    df = pd.read_csv(filepath)
    
    print(f"\nDataset Overview:")
    print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    print(f"\nColumns:")
    for i, col in enumerate(df.columns, 1):
        print(f"{i:2d}. {col}")
    
    print(f"\nData Types:")
    print(df.dtypes)
    
    print(f"\nMissing Values:")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(missing[missing > 0])
    else:
        print("No missing values found")
    
    print(f"\nSample Data (first 3 rows):")
    print(df.head(3))
    
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the shopping trends dataset"""
    
    print(f"\nCleaning dataset...")
    
    # Make a copy to avoid modifying original
    df_clean = df.copy()
    
    # Standardize column names (remove spaces, make lowercase)
    df_clean.columns = [col.strip().lower().replace(' ', '_').replace('(', '').replace(')', '') for col in df_clean.columns]
    
    # Remove any duplicate rows
    initial_rows = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    if len(df_clean) < initial_rows:
        print(f"Removed {initial_rows - len(df_clean)} duplicate rows")
    
    # Basic data validation
    print(f"\nData Validation:")
    print(f"Customer ID range: {df_clean['customer_id'].min()} to {df_clean['customer_id'].max()}")
    print(f"Age range: {df_clean['age'].min()} to {df_clean['age'].max()}")
    print(f"Purchase amount range: ${df_clean['purchase_amount_usd'].min():.2f} to ${df_clean['purchase_amount_usd'].max():.2f}")
    
    # Category distribution
    print(f"\nTop 5 Categories:")
    print(df_clean['category'].value_counts().head())
    
    # Location distribution
    print(f"\nTop 5 Locations:")
    print(df_clean['location'].value_counts().head())
    
    return df_clean

def generate_business_questions(df: pd.DataFrame) -> list:
    """Generate business questions based on the actual dataset"""
    
    questions = [
        "What are the top 5 most popular product categories by purchase volume?",
        "Which age group has the highest average purchase amount?",
        "What is the correlation between customer age and purchase amount?",
        "Which locations have the highest and lowest average purchase amounts?",
        "How does the season affect purchase patterns and amounts?",
        "What is the distribution of payment methods across different customer segments?",
        "Which product categories have the highest and lowest review ratings?",
        "How does the discount application affect purchase amounts?",
        "What is the relationship between previous purchases and current purchase behavior?",
        "Which customer segments are most likely to use promo codes?"
    ]
    
    return questions

def main():
    """Main function to process the shopping trends dataset"""
    
    filepath = "data/shopping_trends.csv"
    
    # Load and analyze
    df = load_and_analyze_data(filepath)
    
    # Clean data
    df_clean = clean_data(df)
    
    # Save cleaned data
    output_path = "data/shopping_trends_clean.csv"
    df_clean.to_csv(output_path, index=False)
    print(f"\nCleaned data saved to: {output_path}")
    
    # Generate business questions
    questions = generate_business_questions(df_clean)
    
    print(f"\nGenerated Business Questions:")
    for i, question in enumerate(questions, 1):
        print(f"{i:2d}. {question}")
    
    # Save questions to file
    import yaml
    questions_data = {"questions": questions}
    with open("data/business_questions.yaml", "w") as f:
        yaml.dump(questions_data, f, default_flow_style=False)
    print(f"\nBusiness questions saved to: data/business_questions.yaml")
    
    print(f"\nData processing completed successfully!")

if __name__ == "__main__":
    main() 