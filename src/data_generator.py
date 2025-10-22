"""
Customer Data Generator Module.

This module generates synthetic customer data using the Faker library.
It creates realistic customer profiles with transaction history, demographics,
and behavioral patterns for clustering analysis.
"""

import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
import logging
from typing import Tuple

from src.config import Config

logger = logging.getLogger(__name__)


class CustomerDataGenerator:
    """Generates synthetic customer data for clustering analysis."""

    def __init__(self, config: Config):
        """
        Initialize the data generator.

        Args:
            config: Configuration object containing generation parameters
        """
        self.config = config
        self.fake = Faker()

        if config.mlops.enable_random_seed:
            Faker.seed(config.data_generator.random_seed)
            np.random.seed(config.data_generator.random_seed)

        logger.info("CustomerDataGenerator initialized")

    def generate_customers(self) -> pd.DataFrame:
        """
        Generate customer demographic data.

        Returns:
            DataFrame with customer profiles
        """
        logger.info(f"Generating {self.config.data_generator.num_customers} customer profiles")

        # Convert string dates to datetime objects
        start_date = datetime.strptime(self.config.data_generator.start_date, '%Y-%m-%d')
        end_date = datetime.strptime(self.config.data_generator.end_date, '%Y-%m-%d')

        customers = []
        for i in range(self.config.data_generator.num_customers):
            customer = {
                'customer_id': f'CUST_{i+1:06d}',
                'name': self.fake.name(),
                'email': self.fake.email(),
                'phone': self.fake.phone_number(),
                'address': self.fake.address().replace('\n', ', '),
                'city': self.fake.city(),
                'state': self.fake.state(),
                'zip_code': self.fake.zipcode(),
                'country': 'USA',
                'registration_date': self.fake.date_between(
                    start_date=start_date,
                    end_date=end_date
                ),
                'age': np.random.randint(18, 80),
                'gender': np.random.choice(['Male', 'Female', 'Other'], p=[0.48, 0.48, 0.04]),
                'income_bracket': np.random.choice(
                    ['Low', 'Medium', 'High', 'Very High'],
                    p=[0.2, 0.4, 0.3, 0.1]
                ),
            }
            customers.append(customer)

        df = pd.DataFrame(customers)
        logger.info(f"Generated {len(df)} customer profiles")
        return df

    def generate_transactions(self, customers_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate transaction history for customers.

        Args:
            customers_df: DataFrame with customer profiles

        Returns:
            DataFrame with transaction records
        """
        logger.info("Generating customer transactions")

        transactions = []
        categories = self.config.data_generator.product_categories
        price_ranges = self.config.data_generator.price_ranges

        for _, customer in customers_df.iterrows():
            customer_id = customer['customer_id']
            reg_date = customer['registration_date']

            # Determine customer behavior segment
            behavior_segment = self._assign_behavior_segment(customer)

            # Generate transactions based on behavior
            num_transactions = self._get_transaction_count(behavior_segment)

            for _ in range(num_transactions):
                transaction_date = self._generate_transaction_date(
                    reg_date,
                    behavior_segment
                )

                category = np.random.choice(categories)
                min_price, max_price = price_ranges[category]

                # Adjust price based on income bracket
                price_multiplier = self._get_price_multiplier(customer['income_bracket'])
                base_price = np.random.uniform(min_price, max_price)
                adjusted_price = base_price * price_multiplier

                quantity = np.random.randint(1, 5)
                total_amount = adjusted_price * quantity

                # Add some discount variation
                discount = np.random.choice([0, 5, 10, 15, 20], p=[0.6, 0.15, 0.15, 0.05, 0.05])
                final_amount = total_amount * (1 - discount/100)

                transaction = {
                    'transaction_id': f'TXN_{len(transactions)+1:08d}',
                    'customer_id': customer_id,
                    'transaction_date': transaction_date,
                    'product_category': category,
                    'product_name': self._generate_product_name(category),
                    'quantity': quantity,
                    'unit_price': adjusted_price,
                    'discount_percent': discount,
                    'total_amount': final_amount,
                    'payment_method': np.random.choice(
                        ['Credit Card', 'Debit Card', 'PayPal', 'Bank Transfer'],
                        p=[0.5, 0.3, 0.15, 0.05]
                    ),
                    'device_type': np.random.choice(
                        ['Desktop', 'Mobile', 'Tablet'],
                        p=[0.4, 0.5, 0.1]
                    ),
                }
                transactions.append(transaction)

        df = pd.DataFrame(transactions)
        logger.info(f"Generated {len(df)} transactions")
        return df

    def _assign_behavior_segment(self, customer: pd.Series) -> str:
        """Assign a behavioral segment to a customer."""
        segments = ['high_value', 'bargain_hunter', 'occasional', 'new', 'at_risk']
        probabilities = [0.15, 0.25, 0.30, 0.20, 0.10]

        # Adjust probabilities based on income
        if customer['income_bracket'] == 'Very High':
            probabilities = [0.40, 0.10, 0.20, 0.20, 0.10]
        elif customer['income_bracket'] == 'Low':
            probabilities = [0.05, 0.40, 0.25, 0.20, 0.10]

        return np.random.choice(segments, p=probabilities)

    def _get_transaction_count(self, behavior_segment: str) -> int:
        """Get number of transactions based on behavior segment."""
        transaction_ranges = {
            'high_value': (15, 50),
            'bargain_hunter': (20, 60),
            'occasional': (3, 15),
            'new': (1, 10),
            'at_risk': (1, 5),
        }
        min_txn, max_txn = transaction_ranges[behavior_segment]
        return np.random.randint(min_txn, max_txn + 1)

    def _generate_transaction_date(self, reg_date, behavior_segment: str):
        """Generate transaction date based on registration date and behavior."""
        end_date = datetime.strptime(self.config.data_generator.end_date, '%Y-%m-%d')

        if behavior_segment == 'at_risk':
            # Generate older transactions
            latest_date = end_date - timedelta(days=180)
        else:
            latest_date = end_date

        # Convert reg_date to datetime if it's a date object
        if hasattr(reg_date, 'hour'):
            # It's already a datetime
            reg_datetime = reg_date
        else:
            # It's a date, convert to datetime
            reg_datetime = datetime.combine(reg_date, datetime.min.time())

        # Ensure start_date is before end_date
        if reg_datetime >= latest_date:
            latest_date = reg_datetime + timedelta(days=1)

        return self.fake.date_between(start_date=reg_datetime, end_date=latest_date)

    def _get_price_multiplier(self, income_bracket: str) -> float:
        """Get price multiplier based on income bracket."""
        multipliers = {
            'Low': 0.7,
            'Medium': 1.0,
            'High': 1.3,
            'Very High': 1.8,
        }
        return multipliers[income_bracket]

    def _generate_product_name(self, category: str) -> str:
        """Generate a realistic product name for a category."""
        products = {
            'Electronics': [
                'Laptop', 'Smartphone', 'Tablet', 'Headphones', 'Smart Watch',
                'Camera', 'Gaming Console', 'Monitor', 'Keyboard', 'Mouse'
            ],
            'Clothing': [
                'T-Shirt', 'Jeans', 'Dress', 'Jacket', 'Sweater',
                'Shoes', 'Sneakers', 'Coat', 'Shirt', 'Pants'
            ],
            'Home & Garden': [
                'Sofa', 'Bed Frame', 'Lamp', 'Rug', 'Curtains',
                'Garden Tools', 'Plants', 'Furniture Set', 'Decor Items', 'Kitchen Set'
            ],
            'Books': [
                'Fiction Novel', 'Non-Fiction', 'Textbook', 'Cookbook', 'Biography',
                'Mystery Novel', 'Self-Help Book', 'Science Fiction', 'History Book', 'Children\'s Book'
            ],
            'Sports': [
                'Running Shoes', 'Yoga Mat', 'Dumbbell Set', 'Bicycle', 'Tennis Racket',
                'Basketball', 'Fitness Tracker', 'Gym Bag', 'Protein Powder', 'Exercise Bike'
            ],
            'Beauty': [
                'Skincare Set', 'Makeup Kit', 'Perfume', 'Hair Products', 'Face Cream',
                'Nail Polish', 'Lipstick', 'Foundation', 'Shampoo', 'Body Lotion'
            ],
            'Toys': [
                'Action Figure', 'Board Game', 'Puzzle', 'Doll', 'Building Blocks',
                'Remote Control Car', 'Educational Toy', 'Stuffed Animal', 'Art Supplies', 'LEGO Set'
            ],
        }
        return np.random.choice(products[category])

    def generate_dataset(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate complete customer dataset with transactions.

        Returns:
            Tuple of (customers_df, transactions_df)
        """
        logger.info("Starting complete dataset generation")

        customers_df = self.generate_customers()
        transactions_df = self.generate_transactions(customers_df)

        logger.info("Dataset generation complete")
        return customers_df, transactions_df

    def merge_and_aggregate(
        self,
        customers_df: pd.DataFrame,
        transactions_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merge customer and transaction data and create aggregate features.

        Args:
            customers_df: Customer profiles DataFrame
            transactions_df: Transactions DataFrame

        Returns:
            Merged DataFrame with aggregate features
        """
        logger.info("Merging and aggregating customer data")

        # Calculate aggregate metrics per customer
        agg_metrics = transactions_df.groupby('customer_id').agg({
            'transaction_id': 'count',
            'total_amount': ['sum', 'mean', 'std'],
            'transaction_date': ['min', 'max'],
            'product_category': 'nunique',
            'discount_percent': 'mean',
        }).reset_index()

        # Flatten column names
        agg_metrics.columns = [
            'customer_id', 'total_transactions', 'total_spent', 'avg_transaction_value',
            'std_transaction_value', 'first_purchase_date', 'last_purchase_date',
            'unique_categories', 'avg_discount_used'
        ]

        # Merge with customer data
        merged_df = customers_df.merge(agg_metrics, on='customer_id', how='left')

        # Fill NaN for customers with no transactions
        merged_df = merged_df.fillna({
            'total_transactions': 0,
            'total_spent': 0,
            'avg_transaction_value': 0,
            'std_transaction_value': 0,
            'unique_categories': 0,
            'avg_discount_used': 0,
        })

        logger.info(f"Merged dataset shape: {merged_df.shape}")
        return merged_df

    def save_data(
        self,
        customers_df: pd.DataFrame,
        transactions_df: pd.DataFrame,
        merged_df: pd.DataFrame
    ):
        """Save generated data to CSV files."""
        data_dir = self.config.paths.data_dir

        customers_df.to_csv(data_dir / 'customers.csv', index=False)
        transactions_df.to_csv(data_dir / 'transactions.csv', index=False)
        merged_df.to_csv(data_dir / 'raw_customer_data.csv', index=False)

        logger.info(f"Data saved to {data_dir}")
