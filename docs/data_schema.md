# Shopping Trends Dataset Schema

## Dataset Overview
- **File**: `shopping_trends.csv`
- **Size**: 3,900 rows × 19 columns
- **Memory**: 3.01 MB
- **Missing Values**: None

## Fields and Data Types

| # | Field Name | Data Type | Description | Range/Values |
|---|------------|-----------|-------------|--------------|
| 1 | Customer ID | int64 | Unique customer identifier | 1 to 3,900 |
| 2 | Age | int64 | Customer age in years | 18 to 70 |
| 3 | Gender | object | Customer gender | Male, Female |
| 4 | Item Purchased | object | Specific product name | Various product names |
| 5 | Category | object | Product category | Clothing, Accessories, Footwear, Outerwear |
| 6 | Purchase Amount (USD) | int64 | Transaction amount | $20 to $100 |
| 7 | Location | object | Customer location (US State) | 50 US states |
| 8 | Size | object | Product size | S, M, L, XL, etc. |
| 9 | Color | object | Product color | Various colors |
| 10 | Season | object | Purchase season | Spring, Summer, Fall, Winter |
| 11 | Review Rating | float64 | Product review rating | 1.0 to 5.0 |
| 12 | Subscription Status | object | Customer subscription | Yes, No |
| 13 | Payment Method | object | Payment method used | Credit Card, PayPal, Cash, etc. |
| 14 | Shipping Type | object | Shipping method | Express, Free Shipping, Next Day Air |
| 15 | Discount Applied | object | Whether discount was applied | Yes, No |
| 16 | Promo Code Used | object | Whether promo code was used | Yes, No |
| 17 | Previous Purchases | int64 | Number of previous purchases | 0 to 50+ |
| 18 | Preferred Payment Method | object | Customer's preferred payment | Various payment methods |
| 19 | Frequency of Purchases | object | Purchase frequency | Weekly, Fortnightly, Monthly |

## Key Insights

### Product Categories
- **Clothing**: 1,737 purchases (44.5%)
- **Accessories**: 1,240 purchases (31.8%)
- **Footwear**: 599 purchases (15.4%)
- **Outerwear**: 324 purchases (8.3%)

### Top Locations
- Montana: 96 purchases
- California: 95 purchases
- Idaho: 93 purchases
- Illinois: 92 purchases
- Alabama: 89 purchases

### Customer Demographics
- Age range: 18-70 years
- Gender distribution: Balanced between Male and Female
- Purchase amounts: $20-$100 range

### Business Patterns
- All customers have subscription status
- Mix of payment methods (Credit Card, PayPal, Cash, Bank Transfer)
- Various shipping options available
- Discount and promo code usage tracked
- Purchase frequency varies (Weekly, Fortnightly, Monthly)

## Data Quality
- ✅ No missing values
- ✅ Consistent data types
- ✅ Logical value ranges
- ✅ No duplicate rows after cleaning

## Use Cases for LLM Analysis
This dataset is ideal for:
1. **Customer segmentation analysis**
2. **Purchase pattern prediction**
3. **Seasonal trend analysis**
4. **Payment method optimization**
5. **Discount strategy evaluation**
6. **Geographic market analysis**
7. **Product category performance**
8. **Customer lifetime value analysis** 