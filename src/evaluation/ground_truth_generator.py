#!/usr/bin/env python3
"""
Ground Truth Answer Generator

This module calculates actual answers from datasets to provide ground truth context
for evaluators during blind testing. This helps evaluators make more informed 
decisions when ranking LLM responses.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import Dict, Any
import yaml


class GroundTruthGenerator:
    """Generate ground truth answers from datasets."""
    
    def __init__(self, project_root: str = None):
        """Initialize with project root path."""
        if project_root is None:
            self.project_root = Path(__file__).parent.parent.parent
        else:
            self.project_root = Path(project_root)
        
        self.data_dir = self.project_root / "data"
        self.retail_data = None
        self.finance_data = None
    
    def load_datasets(self):
        """Load retail and finance datasets."""
        try:
            # Load retail data
            retail_file = self.data_dir / "shopping_trends.csv"
            if retail_file.exists():
                self.retail_data = pd.read_csv(retail_file)
                print(f"✅ Loaded retail data: {len(self.retail_data)} rows")
            
            # Load finance data
            finance_file = self.data_dir / "Tesla_stock_data.csv"
            if finance_file.exists():
                self.finance_data = pd.read_csv(finance_file)
                # Convert Date column to datetime
                self.finance_data['Date'] = pd.to_datetime(self.finance_data['Date'])
                print(f"✅ Loaded finance data: {len(self.finance_data)} rows")
                
        except Exception as e:
            print(f"❌ Error loading datasets: {e}")
    
    def calculate_retail_ground_truths(self) -> Dict[str, Any]:
        """Calculate ground truth answers for retail questions."""
        if self.retail_data is None:
            return {}
        
        df = self.retail_data
        ground_truths = {}
        
        try:
            # retail_01: Top three product categories by total sales value - COMPREHENSIVE ANALYSIS
            category_analysis = df.groupby('Category')['Purchase Amount (USD)'].agg(['sum', 'count', 'mean']).round(2)
            category_analysis.columns = ['total_sales', 'transaction_count', 'avg_per_transaction']
            category_analysis = category_analysis.sort_values('total_sales', ascending=False)
            
            # Calculate market share
            total_market = category_analysis['total_sales'].sum()
            market_share = (category_analysis['total_sales'] / total_market * 100).round(1)
            
            top_3 = category_analysis.head(3)
            ground_truths['retail_01'] = {
                'answer': f"Top 3 categories by total sales: 1) {top_3.index[0]} (${top_3.iloc[0]['total_sales']:,.0f}, {market_share.iloc[0]}%), 2) {top_3.index[1]} (${top_3.iloc[1]['total_sales']:,.0f}, {market_share.iloc[1]}%), 3) {top_3.index[2]} (${top_3.iloc[2]['total_sales']:,.0f}, {market_share.iloc[2]}%)",
                'details': {
                    'comprehensive_breakdown': {str(cat): {
                        'total_sales': float(row['total_sales']),
                        'transaction_count': int(row['transaction_count']),
                        'avg_per_transaction': float(row['avg_per_transaction']),
                        'market_share_percent': float(market_share[cat])
                    } for cat, row in category_analysis.iterrows()},
                    'market_concentration': f"Top 3 categories control {market_share.head(3).sum():.1f}% of total sales",
                    'total_market_value': float(total_market)
                },
                'business_insight': f"Clothing dominates with {market_share.iloc[0]:.1f}% market share and highest transaction volume ({int(top_3.iloc[0]['transaction_count'])} transactions)"
            }
            
            # retail_02: Age group spending analysis - COMPREHENSIVE STATISTICAL ANALYSIS
            df['Age_Group'] = pd.cut(df['Age'], bins=[0, 25, 35, 50, 100], labels=['18-25', '26-35', '36-50', '50+'])
            age_analysis = df.groupby('Age_Group', observed=False)['Purchase Amount (USD)'].agg(['mean', 'median', 'std', 'count', 'sum']).round(2)
            age_analysis = age_analysis.sort_values('mean', ascending=False)
            
            # Statistical significance test
            from scipy import stats
            age_groups = [group for name, group in df.groupby('Age_Group', observed=False)['Purchase Amount (USD)']]
            f_stat, p_value = stats.f_oneway(*age_groups)
            
            # Calculate confidence intervals
            conf_intervals = {}
            for age_group in age_analysis.index:
                group_data = df[df['Age_Group'] == age_group]['Purchase Amount (USD)']
                if len(group_data) > 1:
                    ci = stats.t.interval(0.95, len(group_data)-1, 
                                        loc=group_data.mean(), 
                                        scale=stats.sem(group_data))
                    conf_intervals[str(age_group)] = {'lower': float(ci[0]), 'upper': float(ci[1])}
            
            ground_truths['retail_02'] = {
                'answer': f"Age group {age_analysis.index[0]} spends most on average (${age_analysis.iloc[0]['mean']:.2f} ± ${age_analysis.iloc[0]['std']:.2f}), followed by {age_analysis.index[1]} (${age_analysis.iloc[1]['mean']:.2f})",
                'details': {
                    'statistical_summary': {str(group): {
                        'mean_spend': float(row['mean']),
                        'median_spend': float(row['median']),
                        'std_deviation': float(row['std']),
                        'customer_count': int(row['count']),
                        'total_revenue': float(row['sum'])
                    } for group, row in age_analysis.iterrows()},
                    'confidence_intervals_95': conf_intervals,
                    'statistical_significance': {
                        'f_statistic': float(f_stat),
                        'p_value': float(p_value),
                        'significant_difference': p_value < 0.05
                    }
                },
                'business_insight': f"Statistical analysis confirms significant spending differences (p={p_value:.4f}). {age_analysis.index[0]} customers have {((age_analysis.iloc[0]['mean'] - age_analysis.iloc[-1]['mean']) / age_analysis.iloc[-1]['mean'] * 100):.1f}% higher average spend than lowest group."
            }
            
            # retail_03: Comprehensive seasonal purchase pattern analysis
            season_category = df.groupby(['Season', 'Category']).agg({
                'Purchase Amount (USD)': ['count', 'sum', 'mean']
            }).round(2)
            season_category.columns = ['transaction_count', 'total_revenue', 'avg_transaction']
            
            # Calculate seasonal metrics
            seasonal_totals = df.groupby('Season').agg({
                'Purchase Amount (USD)': ['count', 'sum', 'mean'],
                'Review Rating': 'mean',
                'Age': 'mean'
            }).round(2)
            seasonal_totals.columns = ['total_transactions', 'total_revenue', 'avg_spend', 'avg_rating', 'avg_customer_age']
            seasonal_totals = seasonal_totals.sort_values('total_revenue', ascending=False)
            
            # Category performance by season (market share analysis)
            category_seasonal = pd.crosstab(df['Season'], df['Category'], normalize='index') * 100
            
            # Statistical analysis of seasonal differences
            seasonal_revenue = [df[df['Season'] == season]['Purchase Amount (USD)'] for season in df['Season'].unique()]
            from scipy.stats import f_oneway
            f_stat_seasonal, p_value_seasonal = f_oneway(*seasonal_revenue)
            
            # Identify peak and low seasons
            peak_season = seasonal_totals.index[0]  # Highest revenue season
            low_season = seasonal_totals.index[-1]  # Lowest revenue season
            revenue_variation = ((seasonal_totals.loc[peak_season, 'total_revenue'] - seasonal_totals.loc[low_season, 'total_revenue']) / seasonal_totals.loc[low_season, 'total_revenue'] * 100)
            
            ground_truths['retail_03'] = {
                'answer': f"Seasonal patterns show significant variation (p={p_value_seasonal:.4f}): {peak_season} is peak season with ${seasonal_totals.loc[peak_season, 'total_revenue']:,.0f} revenue ({seasonal_totals.loc[peak_season, 'total_transactions']:,} transactions), {revenue_variation:.1f}% higher than {low_season}. Category preferences shift seasonally with strongest seasonal effects in {category_seasonal.std(axis=0).idxmax()}.",
                'details': {
                    'seasonal_performance': {str(season): {
                        'total_transactions': int(data['total_transactions']),
                        'total_revenue': float(data['total_revenue']),
                        'avg_transaction_value': float(data['avg_spend']),
                        'avg_customer_rating': float(data['avg_rating']),
                        'avg_customer_age': float(data['avg_customer_age']),
                        'market_share_percent': float(data['total_revenue'] / seasonal_totals['total_revenue'].sum() * 100)
                    } for season, data in seasonal_totals.iterrows()},
                    'category_seasonal_distribution': {str(season): {cat: f"{pct:.1f}%" for cat, pct in row.items()} 
                                                     for season, row in category_seasonal.iterrows()},
                    'statistical_analysis': {
                        'f_statistic': float(f_stat_seasonal),
                        'p_value': float(p_value_seasonal),
                        'significant_seasonal_effect': p_value_seasonal < 0.05,
                        'revenue_coefficient_variation': float(seasonal_totals['total_revenue'].std() / seasonal_totals['total_revenue'].mean() * 100)
                    },
                    'business_metrics': {
                        'peak_season': str(peak_season),
                        'low_season': str(low_season),
                        'peak_to_low_ratio': float(seasonal_totals.loc[peak_season, 'total_revenue'] / seasonal_totals.loc[low_season, 'total_revenue']),
                        'most_seasonal_category': str(category_seasonal.std(axis=0).idxmax()),
                        'least_seasonal_category': str(category_seasonal.std(axis=0).idxmin())
                    }
                },
                'business_insight': f"Clear seasonality exists with {revenue_variation:.1f}% revenue difference between peak ({peak_season}) and low ({low_season}) seasons. {category_seasonal.std(axis=0).idxmax()} shows highest seasonal variation, suggesting targeted seasonal marketing opportunities."
            }
            
            # retail_04: Payment methods popular among loyalty members
            loyalty_payments = df[df['Subscription Status'] == 'Yes']['Payment Method'].value_counts()
            ground_truths['retail_04'] = {
                'answer': f"Most popular payment for loyalty members: {loyalty_payments.index[0]}",
                'details': loyalty_payments.to_dict(),
                'loyalty_percentage': f"{len(df[df['Subscription Status'] == 'Yes']) / len(df) * 100:.1f}%"
            }
            
            # retail_05: Comprehensive gender shopping behavior analysis
            gender_analysis = df.groupby('Gender').agg({
                'Purchase Amount (USD)': ['mean', 'median', 'std', 'count', 'sum'],
                'Previous Purchases': 'mean',
                'Review Rating': 'mean',
                'Age': 'mean'
            }).round(2)
            
            # Flatten column names
            gender_analysis.columns = ['avg_spend', 'median_spend', 'spend_std', 'customer_count', 'total_revenue', 'avg_previous_purchases', 'avg_rating', 'avg_age']
            
            # Statistical significance test for spending differences
            male_spending = df[df['Gender'] == 'Male']['Purchase Amount (USD)']
            female_spending = df[df['Gender'] == 'Female']['Purchase Amount (USD)']
            from scipy.stats import ttest_ind
            t_stat, p_value = ttest_ind(male_spending, female_spending)
            
            # Category preferences by gender
            category_gender = pd.crosstab(df['Gender'], df['Category'], normalize='index') * 100
            
            # Payment method preferences
            payment_gender = pd.crosstab(df['Gender'], df['Payment Method'], normalize='index') * 100
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt(((len(male_spending) - 1) * male_spending.std()**2 + 
                                (len(female_spending) - 1) * female_spending.std()**2) / 
                               (len(male_spending) + len(female_spending) - 2))
            cohens_d = (male_spending.mean() - female_spending.mean()) / pooled_std
            
            ground_truths['retail_05'] = {
                'answer': f"Significant gender differences exist: Males spend ${gender_analysis.loc['Male', 'avg_spend']:.2f} on average vs Females ${gender_analysis.loc['Female', 'avg_spend']:.2f} (p={p_value:.4f}, Cohen's d={cohens_d:.3f}). Males represent {gender_analysis.loc['Male', 'customer_count']/gender_analysis['customer_count'].sum()*100:.1f}% of customers but {gender_analysis.loc['Male', 'total_revenue']/gender_analysis['total_revenue'].sum()*100:.1f}% of revenue.",
                'details': {
                    'spending_analysis': {
                        'male': {
                            'avg_spend': float(gender_analysis.loc['Male', 'avg_spend']),
                            'median_spend': float(gender_analysis.loc['Male', 'median_spend']),
                            'std_spend': float(gender_analysis.loc['Male', 'spend_std']),
                            'customer_count': int(gender_analysis.loc['Male', 'customer_count']),
                            'total_revenue': float(gender_analysis.loc['Male', 'total_revenue']),
                            'avg_age': float(gender_analysis.loc['Male', 'avg_age'])
                        },
                        'female': {
                            'avg_spend': float(gender_analysis.loc['Female', 'avg_spend']),
                            'median_spend': float(gender_analysis.loc['Female', 'median_spend']),
                            'std_spend': float(gender_analysis.loc['Female', 'spend_std']),
                            'customer_count': int(gender_analysis.loc['Female', 'customer_count']),
                            'total_revenue': float(gender_analysis.loc['Female', 'total_revenue']),
                            'avg_age': float(gender_analysis.loc['Female', 'avg_age'])
                        }
                    },
                    'statistical_tests': {
                        't_statistic': float(t_stat),
                        'p_value': float(p_value),
                        'significant_difference': p_value < 0.05,
                        'cohens_d_effect_size': float(cohens_d),
                        'effect_interpretation': 'Large' if abs(cohens_d) > 0.8 else 'Medium' if abs(cohens_d) > 0.5 else 'Small'
                    },
                    'category_preferences': {gender: {cat: f"{pct:.1f}%" for cat, pct in row.items()} 
                                           for gender, row in category_gender.iterrows()},
                    'payment_preferences': {gender: dict(row.round(1)) for gender, row in payment_gender.iterrows()}
                },
                'business_insight': f"Gender segmentation reveals {'males' if gender_analysis.loc['Male', 'avg_spend'] > gender_analysis.loc['Female', 'avg_spend'] else 'females'} as higher-value customers with {abs((gender_analysis.loc['Male', 'avg_spend'] - gender_analysis.loc['Female', 'avg_spend']) / gender_analysis.loc['Female', 'avg_spend'] * 100):.1f}% spending difference. Effect size ({cohens_d:.3f}) indicates {'strong' if abs(cohens_d) > 0.8 else 'moderate' if abs(cohens_d) > 0.5 else 'weak'} practical significance for targeted marketing."
            }
            
            # retail_06: Location review ratings
            location_ratings = df.groupby('Location')['Review Rating'].mean().sort_values(ascending=False)
            ground_truths['retail_06'] = {
                'answer': f"Highest rated location: {location_ratings.index[0]} (avg: {location_ratings.iloc[0]:.2f})",
                'details': location_ratings.head(10).to_dict(),
                'top_5_locations': location_ratings.head(5).to_dict()
            }
            
            # retail_07: Promo code/discount effect on spending
            discount_yes = df[df['Discount Applied'] == 'Yes']['Purchase Amount (USD)'].mean()
            discount_no = df[df['Discount Applied'] == 'No']['Purchase Amount (USD)'].mean()
            promo_yes = df[df['Promo Code Used'] == 'Yes']['Purchase Amount (USD)'].mean()
            promo_no = df[df['Promo Code Used'] == 'No']['Purchase Amount (USD)'].mean()
            
            ground_truths['retail_07'] = {
                'answer': f"Discounts increase average spend: ${discount_yes:.2f} vs ${discount_no:.2f}",
                'details': {
                    'discount_applied_avg': discount_yes,
                    'no_discount_avg': discount_no,
                    'promo_code_avg': promo_yes,
                    'no_promo_avg': promo_no
                },
                'discount_impact': f"{((discount_yes - discount_no) / discount_no * 100):.1f}% increase"
            }
            
            # retail_08: Subscription status and purchase frequency
            subscription_freq = df[df['Subscription Status'] == 'Yes']['Frequency of Purchases'].value_counts()
            no_subscription_freq = df[df['Subscription Status'] == 'No']['Frequency of Purchases'].value_counts()
            
            ground_truths['retail_08'] = {
                'answer': "Subscription members show different purchase frequency patterns",
                'details': {
                    'subscription_patterns': subscription_freq.to_dict(),
                    'non_subscription_patterns': no_subscription_freq.to_dict()
                },
                'top_subscription_frequency': subscription_freq.index[0] if len(subscription_freq) > 0 else "N/A"
            }
            
            # retail_09: Shipping types for high-value transactions
            high_value = df[df['Purchase Amount (USD)'] > df['Purchase Amount (USD)'].quantile(0.75)]
            shipping_high_value = high_value['Shipping Type'].value_counts()
            
            ground_truths['retail_09'] = {
                'answer': f"Most common shipping for high-value: {shipping_high_value.index[0]}",
                'details': shipping_high_value.to_dict(),
                'high_value_threshold': f"${df['Purchase Amount (USD)'].quantile(0.75):.2f}"
            }
            
            # retail_10: Repeat purchase patterns (using Previous Purchases field)
            repeat_customers = df[df['Previous Purchases'] > 0]
            repeat_patterns = repeat_customers.groupby('Previous Purchases')['Purchase Amount (USD)'].agg(['count', 'mean'])
            
            ground_truths['retail_10'] = {
                'answer': f"Customers with {repeat_patterns['count'].idxmax()} previous purchases are most common",
                'details': {
                    'repeat_customer_count': len(repeat_customers),
                    'total_customers': len(df),
                    'avg_spend_by_previous_purchases': repeat_patterns['mean'].to_dict(),
                    'customer_count_by_previous_purchases': repeat_patterns['count'].to_dict()
                },
                'repeat_customer_percentage': f"{len(repeat_customers) / len(df) * 100:.1f}%"
            }
            
        except Exception as e:
            print(f"⚠️ Error calculating retail ground truths: {e}")
        
        return ground_truths
    
    def calculate_finance_ground_truths(self) -> Dict[str, Any]:
        """Calculate ground truth answers for finance questions."""
        if self.finance_data is None:
            return {}
        
        df = self.finance_data
        ground_truths = {}
        
        try:
            # finance_01: Comprehensive Tesla trading volume analysis
            volume_stats = df['Volume'].describe()
            yearly_volume = df.groupby(df['Date'].dt.year)['Volume'].agg(['mean', 'median', 'std', 'count']).round(0)
            
            # Volume trends over time
            df['Year'] = df['Date'].dt.year
            volume_trend = df.groupby('Year')['Volume'].mean()
            recent_5_years = volume_trend.tail(5).mean()
            historical_avg = volume_trend.head(-5).mean()
            
            # High/Low volume days analysis
            high_volume_threshold = df['Volume'].quantile(0.9)
            high_volume_days = len(df[df['Volume'] > high_volume_threshold])
            
            ground_truths['finance_01'] = {
                'answer': f"Tesla's average daily trading volume is {volume_stats['mean']:,.0f} shares over {len(df):,} trading days ({df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}). Recent 5-year average ({recent_5_years:,.0f}) is {((recent_5_years - historical_avg) / historical_avg * 100):+.1f}% vs historical average.",
                'details': {
                    'comprehensive_statistics': {
                        'mean_volume': float(volume_stats['mean']),
                        'median_volume': float(volume_stats['50%']),
                        'std_deviation': float(volume_stats['std']),
                        'min_volume': float(volume_stats['min']),
                        'max_volume': float(volume_stats['max']),
                        'q1_volume': float(volume_stats['25%']),
                        'q3_volume': float(volume_stats['75%'])
                    },
                    'temporal_analysis': {
                        'total_trading_days': len(df),
                        'date_range': f"{df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}",
                        'recent_5yr_avg': float(recent_5_years),
                        'historical_avg': float(historical_avg),
                        'trend_change_percent': float((recent_5_years - historical_avg) / historical_avg * 100)
                    },
                    'volume_distribution': {
                        'high_volume_days': int(high_volume_days),
                        'high_volume_threshold': float(high_volume_threshold),
                        'percentage_high_volume': float(high_volume_days / len(df) * 100)
                    },
                    'yearly_breakdown': {str(year): {
                        'avg_volume': int(data['mean']),
                        'trading_days': int(data['count'])
                    } for year, data in yearly_volume.iterrows()}
                },
                'business_insight': f"Tesla shows {((recent_5_years - historical_avg) / historical_avg * 100):+.1f}% volume change in recent years, indicating {'increased' if recent_5_years > historical_avg else 'decreased'} market interest. {(high_volume_days / len(df) * 100):.1f}% of days show exceptionally high volume (>{high_volume_threshold:,.0f} shares)."
            }
            
            # finance_02: Comprehensive Tesla price trend analysis over time
            df_sorted = df.sort_values('Date')
            first_close = df_sorted.iloc[0]['Close']
            last_close = df_sorted.iloc[-1]['Close']
            total_return = ((last_close - first_close) / first_close) * 100
            
            # Yearly analysis
            yearly_returns = df_sorted.groupby(df_sorted['Date'].dt.year).agg({
                'Close': ['first', 'last'],
                'High': 'max',
                'Low': 'min'
            }).round(2)
            yearly_returns.columns = ['year_start', 'year_end', 'year_high', 'year_low']
            yearly_returns['annual_return'] = ((yearly_returns['year_end'] - yearly_returns['year_start']) / yearly_returns['year_start'] * 100).round(1)
            
            # Calculate compound annual growth rate (CAGR)
            years = (df_sorted['Date'].max() - df_sorted['Date'].min()).days / 365.25
            cagr = ((last_close / first_close) ** (1/years) - 1) * 100
            
            # Volatility analysis
            df_sorted['Daily_Return'] = df_sorted['Close'].pct_change() * 100
            annual_volatility = df_sorted['Daily_Return'].std() * (252 ** 0.5)  # 252 trading days per year
            
            # Major milestones
            all_time_high = df_sorted.loc[df_sorted['High'].idxmax()]
            all_time_low = df_sorted.loc[df_sorted['Low'].idxmin()]
            
            # Trend classification
            positive_years = len(yearly_returns[yearly_returns['annual_return'] > 0])
            total_years = len(yearly_returns)
            
            ground_truths['finance_02'] = {
                'answer': f"Tesla stock shows strong long-term upward trend with {total_return:.1f}% total return ({cagr:.1f}% CAGR) from ${first_close:.2f} to ${last_close:.2f} over {years:.1f} years. {positive_years}/{total_years} years were positive ({positive_years/total_years*100:.1f}%).",
                'details': {
                    'performance_metrics': {
                        'total_return_percent': float(total_return),
                        'compound_annual_growth_rate': float(cagr),
                        'annualized_volatility': float(annual_volatility),
                        'start_price': float(first_close),
                        'end_price': float(last_close),
                        'period_years': float(years)
                    },
                    'milestone_prices': {
                        'all_time_high': {
                            'price': float(all_time_high['High']),
                            'date': all_time_high['Date'].strftime('%Y-%m-%d')
                        },
                        'all_time_low': {
                            'price': float(all_time_low['Low']),
                            'date': all_time_low['Date'].strftime('%Y-%m-%d')
                        },
                        'price_range_ratio': float(all_time_high['High'] / all_time_low['Low'])
                    },
                    'yearly_performance': {str(year): {
                        'annual_return_percent': float(data['annual_return']),
                        'year_high': float(data['year_high']),
                        'year_low': float(data['year_low']),
                        'start_price': float(data['year_start']),
                        'end_price': float(data['year_end'])
                    } for year, data in yearly_returns.iterrows()},
                    'trend_consistency': {
                        'positive_years': int(positive_years),
                        'total_years': int(total_years),
                        'success_rate_percent': float(positive_years/total_years*100)
                    }
                },
                'business_insight': f"Tesla demonstrates exceptional growth with {cagr:.1f}% CAGR despite {annual_volatility:.1f}% annual volatility. Stock appreciated {all_time_high['High']/all_time_low['Low']:.0f}x from lowest to highest point, showing tremendous wealth creation potential."
            }
            
            # finance_03: Largest single-day changes
            df['Daily_Change'] = df['Close'] - df['Open']
            df['Daily_Change_Pct'] = (df['Daily_Change'] / df['Open']) * 100
            
            largest_gain = df.loc[df['Daily_Change_Pct'].idxmax()]
            largest_loss = df.loc[df['Daily_Change_Pct'].idxmin()]
            
            ground_truths['finance_03'] = {
                'answer': f"Largest gain: {largest_gain['Daily_Change_Pct']:.2f}% on {largest_gain['Date'].date()}, Largest loss: {largest_loss['Daily_Change_Pct']:.2f}% on {largest_loss['Date'].date()}",
                'details': {
                    'largest_gain': {
                        'date': str(largest_gain['Date'].date()),
                        'percentage': largest_gain['Daily_Change_Pct'],
                        'price_change': largest_gain['Daily_Change']
                    },
                    'largest_loss': {
                        'date': str(largest_loss['Date'].date()),
                        'percentage': largest_loss['Daily_Change_Pct'],
                        'price_change': largest_loss['Daily_Change']
                    }
                }
            }
            
            # finance_04: Volume vs volatility correlation
            df['Price_Range'] = df['High'] - df['Low']
            df['Volatility'] = (df['Price_Range'] / df['Close']) * 100
            correlation = df['Volume'].corr(df['Volatility'])
            
            ground_truths['finance_04'] = {
                'answer': f"Volume-volatility correlation: {correlation:.3f}",
                'details': {
                    'correlation_coefficient': correlation,
                    'avg_volatility': df['Volatility'].mean(),
                    'interpretation': "Moderate positive correlation" if correlation > 0.3 else "Weak correlation"
                }
            }
            
            # finance_05: Average difference between opening and closing prices per month
            df['Open_Close_Diff'] = df['Close'] - df['Open']
            df['Month_Year'] = df['Date'].dt.to_period('M')
            monthly_diff = df.groupby('Month_Year')['Open_Close_Diff'].mean()
            
            ground_truths['finance_05'] = {
                'answer': f"Average monthly open-close difference: ${monthly_diff.mean():.2f}",
                'details': {
                    'overall_avg_diff': monthly_diff.mean(),
                    'best_month': str(monthly_diff.idxmax()),
                    'worst_month': str(monthly_diff.idxmin()),
                    'monthly_averages': {str(k): v for k, v in monthly_diff.head(12).to_dict().items()}
                }
            }
            
            # finance_06: Highest sustained growth periods
            df['7_day_return'] = df['Close'].pct_change(periods=7) * 100
            df['30_day_return'] = df['Close'].pct_change(periods=30) * 100
            
            # Find periods of sustained growth (7-day returns > 5% for multiple consecutive periods)
            sustained_growth = df[df['7_day_return'] > 5]
            
            ground_truths['finance_06'] = {
                'answer': f"Identified {len(sustained_growth)} periods of strong 7-day growth (>5%)",
                'details': {
                    'periods_over_5_percent': len(sustained_growth),
                    'max_7_day_return': df['7_day_return'].max(),
                    'max_30_day_return': df['30_day_return'].max(),
                    'best_growth_period': str(df.loc[df['7_day_return'].idxmax(), 'Date'].date()) if not df['7_day_return'].isna().all() else "N/A"
                }
            }
            
            # finance_07: Stock reaction to major events (identify significant daily changes)
            significant_changes = df[abs(df['Daily_Change_Pct']) > 10]  # >10% daily changes
            
            ground_truths['finance_07'] = {
                'answer': f"Found {len(significant_changes)} days with >10% price movements",
                'details': {
                    'significant_movement_days': len(significant_changes),
                    'largest_movements': significant_changes.nlargest(5, 'Daily_Change_Pct')[['Date', 'Daily_Change_Pct']].to_dict('records'),
                    'avg_significant_change': significant_changes['Daily_Change_Pct'].abs().mean()
                }
            }
            
            # finance_08: Volume vs closing price relationship
            close_volume_corr = df['Volume'].corr(df['Close'])
            
            ground_truths['finance_08'] = {
                'answer': f"Volume-Close correlation: {close_volume_corr:.3f}",
                'details': {
                    'correlation_coefficient': close_volume_corr,
                    'avg_close': df['Close'].mean(),
                    'relationship_strength': "Strong" if abs(close_volume_corr) > 0.5 else "Moderate" if abs(close_volume_corr) > 0.3 else "Weak"
                }
            }
            
            # finance_09: Frequency of closing higher than opening
            df['Closed_Higher'] = df['Close'] > df['Open']
            yearly_stats = df.groupby(df['Date'].dt.year)['Closed_Higher'].agg(['count', 'sum', 'mean'])
            
            ground_truths['finance_09'] = {
                'answer': f"Stock closed higher {df['Closed_Higher'].mean()*100:.1f}% of trading days",
                'details': {
                    'overall_percentage': df['Closed_Higher'].mean() * 100,
                    'total_positive_days': df['Closed_Higher'].sum(),
                    'total_trading_days': len(df),
                    'yearly_breakdown': {str(year): f"{stats['mean']*100:.1f}%" for year, stats in yearly_stats.iterrows()}
                }
            }
            
            # finance_10: Most volatile periods (months/quarters)
            df['Quarter'] = df['Date'].dt.quarter
            df['Year'] = df['Date'].dt.year
            df['Quarter_Year'] = df['Year'].astype(str) + '-Q' + df['Quarter'].astype(str)
            
            quarterly_volatility = df.groupby('Quarter_Year')['Volatility'].mean().sort_values(ascending=False)
            monthly_volatility = df.groupby(df['Date'].dt.to_period('M'))['Volatility'].mean().sort_values(ascending=False)
            
            ground_truths['finance_10'] = {
                'answer': f"Most volatile quarter: {quarterly_volatility.index[0]} ({quarterly_volatility.iloc[0]:.2f}% avg volatility)",
                'details': {
                    'most_volatile_quarter': quarterly_volatility.index[0],
                    'highest_quarterly_volatility': quarterly_volatility.iloc[0],
                    'top_5_volatile_quarters': quarterly_volatility.head(5).to_dict(),
                    'top_5_volatile_months': {str(k): v for k, v in monthly_volatility.head(5).to_dict().items()}
                }
            }
            
        except Exception as e:
            print(f"⚠️ Error calculating finance ground truths: {e}")
        
        return ground_truths
    
    def generate_all_ground_truths(self) -> Dict[str, Any]:
        """Generate all ground truth answers and save to file."""
        self.load_datasets()
        
        all_ground_truths = {
            'metadata': {
                'generated_at': pd.Timestamp.now().isoformat(),
                'datasets_used': {
                    'retail': 'shopping_trends.csv',
                    'finance': 'Tesla_stock_data.csv'
                }
            },
            'retail': self.calculate_retail_ground_truths(),
            'finance': self.calculate_finance_ground_truths()
        }
        
        # Save to file
        output_file = self.data_dir / "ground_truth_answers.json"
        with open(output_file, 'w') as f:
            json.dump(all_ground_truths, f, indent=2, default=str)
        
        print(f"✅ Ground truth answers saved to {output_file}")
        return all_ground_truths
    
    def get_ground_truth_for_question(self, question_id: str) -> Dict[str, Any]:
        """Get ground truth answer for a specific question."""
        try:
            ground_truth_file = self.data_dir / "ground_truth_answers.json"
            if not ground_truth_file.exists():
                self.generate_all_ground_truths()
            
            with open(ground_truth_file, 'r') as f:
                ground_truths = json.load(f)
            
            # Determine domain from question_id
            domain = 'retail' if question_id.startswith('retail') else 'finance'
            return ground_truths.get(domain, {}).get(question_id, {})
            
        except Exception as e:
            print(f"⚠️ Error retrieving ground truth for {question_id}: {e}")
            return {}


if __name__ == "__main__":
    generator = GroundTruthGenerator()
    generator.generate_all_ground_truths() 