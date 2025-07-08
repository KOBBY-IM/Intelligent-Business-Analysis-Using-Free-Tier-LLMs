#!/usr/bin/env python3
"""
Test RAG pipeline with retail data using unified LLM providers
"""

import logging
import os
import sys

import pandas as pd

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from llm_providers.groq_provider import GroqProvider
from rag.pipeline import RAGPipeline
from rag.vector_store import FAISSVectorStore

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_retail_data(data_path: str) -> list:
    """
    Load and prepare retail data for RAG

    Args:
        data_path: Path to the CSV file

    Returns:
        List of documents for vector store
    """
    logger.info(f"Loading retail data from {data_path}")

    # Load CSV data
    df = pd.read_csv(data_path)

    # Create documents from the data
    documents = []

    # Document 1: Dataset overview
    overview_text = f"""
    Retail Shopping Trends Dataset Overview:\n\nThis dataset contains {len(df)} records of retail shopping data.\nThe dataset includes the following columns: {', '.join(df.columns.tolist())}\n"""
    if "Date" in df.columns:
        overview_text += f"- Date range: {df['Date'].min()} to {df['Date'].max()}\n"
    if "Product_Category" in df.columns:
        overview_text += f"- Product categories: {df['Product_Category'].nunique()} unique categories\n"
    if "Customer_Segment" in df.columns:
        overview_text += (
            f"- Customer segments: {df['Customer_Segment'].nunique()} segments\n"
        )
    overview_text += (
        f"- Total records: {len(df)}\n- Number of columns: {len(df.columns)}\n"
    )

    documents.append(
        {
            "text": overview_text,
            "source": "dataset_overview",
            "metadata": {
                "type": "overview",
                "rows": len(df),
                "columns": len(df.columns),
            },
        }
    )

    # Document 2: Product category analysis
    if "Product_Category" in df.columns:
        category_stats = df["Product_Category"].value_counts()
        category_text = "Product Category Analysis:\n\n"
        for category, count in category_stats.head(10).items():
            percentage = (count / len(df)) * 100
            category_text += f"- {category}: {count} sales ({percentage:.1f}%)\n"
        documents.append(
            {
                "text": category_text,
                "source": "product_categories",
                "metadata": {"type": "category_analysis", "top_categories": 10},
            }
        )
    else:
        logger.warning(
            "Column 'Product_Category' not found. Skipping category analysis."
        )

    # Document 3: Customer segment analysis
    if "Customer_Segment" in df.columns:
        segment_stats = df["Customer_Segment"].value_counts()
        segment_text = "Customer Segment Analysis:\n\n"
        for segment, count in segment_stats.items():
            percentage = (count / len(df)) * 100
            segment_text += f"- {segment}: {count} customers ({percentage:.1f}%)\n"
        documents.append(
            {
                "text": segment_text,
                "source": "customer_segments",
                "metadata": {"type": "segment_analysis"},
            }
        )
    else:
        logger.warning(
            "Column 'Customer_Segment' not found. Skipping segment analysis."
        )

    # Document 4: Sales trends by month
    if "Date" in df.columns and "Sales_Amount" in df.columns:
        try:
            df["Month"] = pd.to_datetime(df["Date"]).dt.month
            monthly_sales = (
                df.groupby("Month")["Sales_Amount"]
                .agg(["sum", "count", "mean"])
                .round(2)
            )
            monthly_text = "Monthly Sales Trends:\n\n"
            for month, stats in monthly_sales.iterrows():
                monthly_text += f"Month {month}: Total Sales ${stats['sum']:,.2f}, {stats['count']} transactions, Avg ${stats['mean']:.2f}\n"
            documents.append(
                {
                    "text": monthly_text,
                    "source": "monthly_trends",
                    "metadata": {"type": "monthly_analysis"},
                }
            )
        except Exception as e:
            logger.warning(f"Failed to compute monthly sales trends: {e}")
    else:
        logger.warning(
            "Columns 'Date' or 'Sales_Amount' not found. Skipping monthly sales trends."
        )

    # Document 5: Top performing products
    if "Product_Name" in df.columns and "Sales_Amount" in df.columns:
        product_sales = (
            df.groupby("Product_Name")["Sales_Amount"]
            .sum()
            .sort_values(ascending=False)
        )
        product_text = "Top Performing Products:\n\n"
        for product, sales in product_sales.head(10).items():
            product_text += f"- {product}: ${sales:,.2f}\n"
        documents.append(
            {
                "text": product_text,
                "source": "top_products",
                "metadata": {"type": "product_analysis", "top_products": 10},
            }
        )
    else:
        logger.warning(
            "Columns 'Product_Name' or 'Sales_Amount' not found. Skipping top products analysis."
        )

    logger.info(f"Created {len(documents)} documents from retail data")
    return documents


def test_rag_pipeline():
    """Test the complete RAG pipeline with retail data"""

    # Data path
    data_path = "data/shopping_trends.csv"

    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        return

    # Load retail data
    documents = load_retail_data(data_path)

    # Initialize vector store
    logger.info("Initializing FAISS vector store...")
    vector_store = FAISSVectorStore(
        model_name="all-MiniLM-L6-v2", chunk_size=300, chunk_overlap=50
    )

    # Add documents to vector store
    vector_store.add_documents(documents)

    # Initialize Groq provider
    logger.info("Initializing Groq provider...")
    try:
        groq_provider = GroqProvider()

        # Test health check
        if not groq_provider.health_check():
            logger.error("Failed to connect to Groq API")
            return
        logger.info("Groq connection successful")

    except Exception as e:
        logger.error(f"Failed to initialize Groq provider: {e}")
        return

    # Initialize RAG pipeline
    logger.info("Initializing RAG pipeline...")
    pipeline = RAGPipeline(
        vector_store=vector_store,
        llm_provider=groq_provider,
        top_k=5,
        max_context_length=2000,
    )

    # Test queries
    test_queries = [
        "What are the top performing product categories?",
        "How do sales vary by customer segment?",
        "What are the monthly sales trends?",
        "Which products generate the highest revenue?",
        "What is the overall structure of this retail dataset?",
    ]

    logger.info("=" * 80)
    logger.info("TESTING RAG PIPELINE WITH RETAIL DATA")
    logger.info("=" * 80)

    results = []

    for i, query in enumerate(test_queries, 1):
        logger.info(f"\nQuery {i}: {query}")
        logger.info("-" * 60)

        try:
            result = pipeline.query(query, max_tokens=300)
            results.append(result)

            # Log results
            logger.info(f"Response: {result.response}")
            logger.info(f"Total time: {result.total_time_ms:.2f}ms")
            logger.info(
                f"Retrieval time: {result.pipeline_metrics['retrieval_time_ms']:.2f}ms"
            )
            logger.info(
                f"LLM time: {result.pipeline_metrics['llm_generation_time_ms']:.2f}ms"
            )
            logger.info(
                f"Context length: {result.pipeline_metrics['context_length']} chars"
            )
            logger.info(
                f"Chunks retrieved: {result.pipeline_metrics['num_chunks_retrieved']}"
            )

            # Log retrieval quality
            quality = result.pipeline_metrics["retrieval_quality"]
            logger.info(
                f"Retrieval quality - Avg similarity: {quality['avg_similarity']:.3f}, "
                f"Diversity: {quality['diversity']:.3f}"
            )

        except Exception as e:
            logger.error(f"Error processing query {i}: {e}")

    # Summary statistics
    logger.info("\n" + "=" * 80)
    logger.info("PIPELINE PERFORMANCE SUMMARY")
    logger.info("=" * 80)

    successful_results = [r for r in results if r.llm_response.success]

    if successful_results:
        avg_total_time = sum(r.total_time_ms for r in successful_results) / len(
            successful_results
        )
        avg_retrieval_time = sum(
            r.pipeline_metrics["retrieval_time_ms"] for r in successful_results
        ) / len(successful_results)
        avg_llm_time = sum(
            r.pipeline_metrics["llm_generation_time_ms"] for r in successful_results
        ) / len(successful_results)

        logger.info(
            f"Successful queries: {len(successful_results)}/{len(test_queries)}"
        )
        logger.info(f"Average total time: {avg_total_time:.2f}ms")
        logger.info(f"Average retrieval time: {avg_retrieval_time:.2f}ms")
        logger.info(f"Average LLM time: {avg_llm_time:.2f}ms")

        # Check performance criteria
        if avg_total_time < 10000:  # 10 seconds
            logger.info("✅ Performance target met: Response times under 10 seconds")
        else:
            logger.warning(
                "⚠️ Performance target not met: Response times over 10 seconds"
            )

    # Save pipeline for future use
    save_dir = "data/rag_pipeline"
    pipeline.save_pipeline(save_dir)
    logger.info(f"Pipeline saved to {save_dir}")

    return results


if __name__ == "__main__":
    test_rag_pipeline()
